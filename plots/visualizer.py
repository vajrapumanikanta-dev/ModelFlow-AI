"""
Visualizations Module (plots/visualizer.py)
Generates high-quality Matplotlib and Seaborn charts for Dataset Profiling,
Classification, Regression, Clustering, and Explainability.
"""

import io
import base64
from typing import List, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set default aesthetic style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def fig_to_base64(fig: plt.Figure) -> str:
    """Converts a matplotlib Figure to a base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


# 1. Dataset Profiling Visualizations

def plot_distribution(df: pd.DataFrame, col: str) -> plt.Figure:
    """Plots histogram with KDE for numeric column, or countplot for categorical."""
    fig, ax = plt.subplots(figsize=(7, 4))
    series = df[col].dropna()
    
    if pd.api.types.is_numeric_dtype(series):
        sns.histplot(series, kde=True, ax=ax, color="#3b82f6", bins=30)
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    else:
        top_cats = series.value_counts().head(15)
        sns.barplot(x=top_cats.values, y=top_cats.index.astype(str), ax=ax, color="#6366f1")
        ax.set_title(f"Top Categories in {col}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        
    fig.tight_layout()
    return fig


def plot_boxplot(df: pd.DataFrame, col: str) -> plt.Figure:
    """Plots boxplot with outlier markers for a numeric column."""
    fig, ax = plt.subplots(figsize=(7, 3))
    series = df[col].dropna()
    sns.boxplot(x=series, ax=ax, color="#10b981", fliersize=4)
    ax.set_title(f"Box Plot of {col}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(col)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> Optional[plt.Figure]:
    """Plots correlation matrix heatmap for numeric columns."""
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr()
    fig_size = max(6, min(12, len(numeric_cols) * 0.75))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    
    sns.heatmap(
        corr,
        annot=len(numeric_cols) <= 12,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str] = None) -> plt.Figure:
    """Plots 2D scatter plot between two columns."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, ax=ax, alpha=0.7, palette="tab10" if hue else None)
    ax.set_title(f"{x_col} vs {y_col}", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def plot_missing_values_bar(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Plots missing value counts and percentages across columns."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None

    missing_pct = (missing / len(df)) * 100
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(missing) * 0.35)))
    sns.barplot(x=missing_pct.values, y=missing_pct.index.astype(str), ax=ax, color="#ef4444")
    ax.set_title("Missing Values by Column (%)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Missing Percentage (%)")
    ax.set_ylabel("Column")
    
    for i, v in enumerate(missing_pct.values):
        ax.text(v + 0.5, i, f"{v:.1f}% ({missing.values[i]})", va="center", fontsize=9)
        
    ax.set_xlim(0, max(missing_pct.values) + 15)
    fig.tight_layout()
    return fig


# 2. Classification Visualizations

def plot_confusion_matrix_heatmap(
    cm: Union[List[List[int]], np.ndarray],
    class_names: Optional[List[str]] = None,
) -> plt.Figure:
    """Plots annotated confusion matrix."""
    cm_arr = np.asarray(cm)
    n_classes = cm_arr.shape[0]
    if not class_names or len(class_names) != n_classes:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
    )
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_roc_curve_chart(fpr: List[float], tpr: List[float], roc_auc: Optional[float] = None) -> plt.Figure:
    """Plots ROC curve with AUC metric."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    auc_label = f"ROC Curve (AUC = {roc_auc:.3f})" if roc_auc is not None else "ROC Curve"
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=auc_label)
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_precision_recall_curve_chart(precision: List[float], recall: List[float]) -> plt.Figure:
    """Plots Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color="#059669", lw=2.5, label="Precision-Recall")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Plots horizontal bar chart of top N feature importances."""
    df_top = importance_df.head(top_n).sort_values(by="abs_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(df_top) * 0.3)))
    
    sns.barplot(
        data=df_top,
        x="abs_importance",
        y="feature",
        ax=ax,
        color="#8b5cf6",
    )
    ax.set_title(f"Top {len(df_top)} Feature Importances", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Importance Magnitude")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


# 3. Regression Visualizations

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Plots Actual vs Predicted values with a 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, color="#0284c7", edgecolors="none")
    
    # 45 degree perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", lw=2, label="Ideal Fit (y=x)")
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Values", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Plots residual errors against predicted values."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, color="#f59e0b", edgecolors="none")
    ax.axhline(0, color="#ef4444", linestyle="--", lw=1.5)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# 4. Clustering & Dimensionality Reduction Visualizations

def plot_clustering_elbow_chart(k_range: List[int], inertias: List[float], recommended_k: int = 3) -> plt.Figure:
    """Plots Elbow method inertia vs K."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(k_range, inertias, marker="o", color="#3b82f6", lw=2)
    if recommended_k in k_range:
        idx = k_range.index(recommended_k)
        ax.scatter([recommended_k], [inertias[idx]], color="#ef4444", s=100, zorder=5, label=f"Optimal k={recommended_k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_title("Elbow Method for Optimal k", fontsize=12, fontweight="bold", pad=10)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_clustering_silhouette_chart(k_range: List[int], silhouette_scores: List[float]) -> plt.Figure:
    """Plots Silhouette Scores across cluster numbers k."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(k_range, silhouette_scores, marker="s", color="#10b981", lw=2)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs Number of Clusters", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def plot_pca_2d_scatter(X_pca: np.ndarray, labels: Optional[Union[List[Any], np.ndarray]] = None) -> plt.Figure:
    """Plots 2D projection using PCA."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if labels is not None:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", ax=ax, alpha=0.8)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], color="#3b82f6", alpha=0.7)
        
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA 2D Cluster Projection", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig

# ML Engineer Assistant

An intelligent ML platform that reduces repetitive manual work in the machine learning workflow, from dataset upload and analysis to model training, evaluation, explainability, prediction, and an automatically generated report.

> **Scope:** Upload → Report. Deployment, model versioning, and monitoring are out of scope for now.

---

## Table of Contents

1. [Features](#features)
2. [Pipeline](#pipeline)
3. [Important design rule: avoid data leakage](#important-design-rule-avoid-data-leakage)
4. [Modules](#modules)
5. [Module details](#module-details)
6. [Tech stack](#tech-stack)
7. [Project structure](#project-structure)
8. [Build phases](#build-phases)
9. [Timeline](#timeline)
10. [Testing](#testing)
11. [Getting started](#getting-started)

---

## Features

- Upload CSV, Excel, and JSON datasets with automatic file-type detection
- Automatic data profiling (rows, columns, dtypes, missing values, duplicates, statistics)
- Interactive, user-controlled data cleaning (the user decides how to handle each issue)
- Feature and target selection with automatic problem-type detection
- Data quality checks: leakage detection and class imbalance detection
- Preprocessing with scikit-learn pipelines
- Train and compare multiple models with cross-validation
- Hyperparameter tuning
- Evaluation metrics and automatic graphs
- Explainability: feature importance, SHAP, LIME
- Save the best model, run single and batch predictions
- Experiment tracking
- Automatic PDF/HTML ML Analysis Report

---

## Pipeline

```
Upload → Detect file type → Load → Profile → Clean (user-controlled)
  → Select features & target → Detect problem type
  → Data-quality checks (leakage, imbalance)
  → Train/test split → Preprocessing pipeline
  → Train multiple models (+ cross-validation)
  → Hyperparameter tuning → Evaluation → Graphs
  → Feature importance + SHAP/LIME → Model comparison
  → Save model → Single / batch prediction
  → Experiment tracking → Generate report
```

---

## Important design rule: avoid data leakage

Split the data **before** fitting any imputer, scaler, or encoder. Fitting them on the full dataset leaks test information into training and inflates scores.

| Safe on the full dataset | Must be fitted on the training set only |
|---|---|
| Removing duplicates | Imputation (mean / median / mode) |
| Removing constant columns | Scaling |
| Removing ID columns | Encoding |
| Fixing invalid values | Outlier capping |
| | SMOTE and other resampling |

Build preprocessing as one scikit-learn `Pipeline` with `ColumnTransformer`. Use `imblearn.pipeline.Pipeline` when SMOTE is involved so it is applied only inside training folds.

---

## Modules

| # | Module | Description |
|---|---|---|
| 1 | Ingestion | CSV, Excel, JSON; type detection; encoding fallback; size limit |
| 2 | Profiling | Rows, columns, dtypes, missing %, duplicates, unique counts, min/max/mean/median, memory usage |
| 3 | Cleaning | Missing values, duplicates, outliers (IQR / Z-score), constant columns, ID columns, with a cleaning log |
| 4 | Feature and target selection | Checkbox for features, dropdown for target, warnings for ID columns |
| 5 | Problem detection | Supervised vs unsupervised; binary, multiclass, or regression guessed from the target |
| 6 | Data quality checks | Leakage detection, class imbalance detection |
| 7 | Preprocessing | Scaling, encoding, train/test split, pipeline |
| 8 | Training | Multiple models plus k-fold cross-validation |
| 9 | Tuning | GridSearchCV / RandomizedSearchCV |
| 10 | Evaluation and graphs | Metrics, leaderboard, plots |
| 11 | Explainability | Feature importance, SHAP, LIME |
| 12 | Model saving and prediction | joblib, single prediction, batch prediction |
| 13 | Experiment tracking | Save and compare runs |
| 14 | Report generation | PDF/HTML ML Analysis Report |

---

## Module details

### Data cleaning (user stays in control)

The app never modifies data silently. For each issue it shows the detected problem and lets the user choose.

| Issue | Options |
|---|---|
| Missing values | Remove rows, mean, median, mode, forward fill |
| Duplicates | Remove, keep |
| Outliers | Keep, remove, cap (IQR or Z-score) |
| Constant columns | Remove, keep |
| ID columns | Exclude, keep |
| Encoding | One-hot, ordinal, label (with a recommendation) |
| Scaling | StandardScaler, MinMaxScaler, RobustScaler, none |

Every choice is stored in a cleaning log so it can be replayed, undone, and included in the report.

### Problem detection

- Target is text/boolean, or numeric with few unique values (for example 10 or fewer, or under about 5% of rows) → **classification**
- Otherwise numeric → **regression**
- No target selected → **unsupervised** (clustering, PCA)
- The user can always override the detected type.

### Leakage detection (heuristics)

- Feature name similar to the target name (for example `Churn` vs `Churn_Status`)
- Very high correlation with the target (above about 0.95)
- A single feature that predicts the target almost perfectly
- Suspicious names such as `final`, `outcome`, `result`, `post_`, `future_`

The app shows a warning and lets the user exclude the column.

### Class imbalance

Flag when the minority class is below about 20%. Offer class weights, random oversampling, SMOTE, or undersampling. Use stratified splits and report F1 and ROC-AUC, not accuracy alone.

### Models

| Task | Models |
|---|---|
| Classification | Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM, Gradient Boosting |
| Regression | Linear, Ridge, Lasso, Decision Tree, Random Forest, SVR, Gradient Boosting |
| Clustering | K-Means (elbow and silhouette), DBSCAN, Hierarchical |
| Dimensionality reduction | PCA |

XGBoost, LightGBM, and CatBoost can be added later.

### Metrics

| Task | Metrics |
|---|---|
| Classification | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| Regression | MAE, MSE, RMSE, R², adjusted R² |
| Clustering | Silhouette score |

Cross-validation (for example 5-fold) is used for more reliable estimates than a single split.

### Visualization

- **Dataset:** histogram, box plot, bar chart, correlation heatmap, scatter plot, missing value chart
- **Classification:** confusion matrix, ROC curve, precision-recall curve, feature importance, class distribution
- **Regression:** actual vs predicted, residual plot, feature importance, prediction error
- **Clustering:** cluster plot, elbow method, silhouette score, PCA 2D plot

### Explainability

- Feature importance for tree-based models
- SHAP for global and per-prediction explanations
- LIME for local explanations

### Prediction

- The saved object must be the **whole pipeline** (preprocessing plus model), not just the model, so raw input works.
- **Single prediction:** form generated automatically from the training columns, with prediction and probability.
- **Batch prediction:** upload a new file, validate that columns match, predict, and download `results.csv`.

### Experiment tracking

For each run, store: dataset, model, selected features, parameters, metrics, and timestamp. Allow saving and comparing experiments. Start with JSON or SQLite; MLflow can be added later.

### Report contents

1. Dataset information (rows, columns, data types, missing values)
2. Cleaning performed (the cleaning log)
3. Selected features and target
4. ML problem type
5. Models trained
6. Results table (accuracy, precision, recall, F1, or regression metrics)
7. Best model
8. Key graphs
9. Feature importance and SHAP
10. Experiment parameters

---

## Tech stack

| Layer | Tools |
|---|---|
| UI | Streamlit (fastest start), or Flask with HTML/JS and Plotly.js |
| ML | pandas, NumPy, scikit-learn, imbalanced-learn, joblib |
| Explainability | SHAP, LIME |
| Graphs | Matplotlib, Seaborn, Plotly |
| Report | ReportLab, WeasyPrint, or HTML with Jinja2 |
| Experiment tracking | JSON / SQLite (MLflow later) |

Keep the ML logic in separate service modules and the UI layer thin, so the interface can be swapped later without rewriting the core.

---

## Project structure

```
ml-assistant/
├── app.py                 # UI entry point
├── core/
│   ├── loader.py          # detect type, read file
│   ├── profiler.py        # dataset and column statistics
│   ├── cleaner.py         # missing, duplicates, outliers, etc.
│   ├── quality.py         # leakage and imbalance detection
│   ├── preprocess.py      # pipeline builder
│   ├── models.py          # model registry
│   ├── trainer.py         # training, CV, tuning
│   ├── evaluator.py       # metrics
│   ├── explain.py         # SHAP, LIME
│   ├── predictor.py       # single and batch prediction
│   ├── tracker.py         # experiment tracking
│   └── report.py          # report generation
├── plots/
├── saved_models/
├── experiments/
├── reports/
├── tests/
├── requirements.txt
└── README.md
```

---

## Build phases

### Phase 1: MVP

Upload → profile → clean → feature/target selection → one problem type → train a few models → evaluate → basic graphs.

### Phase 2: Strong ML project

Multiple models, cross-validation, hyperparameter tuning, feature importance, imbalance handling, leakage detection, model comparison dashboard, unsupervised learning and PCA.

### Phase 3: Advanced features

SHAP/LIME, model saving, single and batch prediction, experiment tracking, automatic report.

> Do not try to build every module at once. Finish each phase before starting the next.

---

## Timeline

About 10 weeks, part-time.

| Weeks | Focus |
|---|---|
| 1 | Setup, upload, loader, profiling |
| 2 to 3 | Cleaning engine and cleaning UI |
| 4 | Feature/target selection, problem detection, preprocessing pipeline |
| 5 | Training, cross-validation, evaluation, leaderboard |
| 6 | Graphs, feature importance |
| 7 | Leakage and imbalance detection, tuning |
| 8 | Unsupervised learning (clustering, PCA) |
| 9 | SHAP/LIME, model saving, single and batch prediction |
| 10 | Experiment tracking, report generation, testing, README and demo |

---

## Testing

- **Test datasets:** Titanic, Iris, Telco Churn, California Housing
- **Edge cases:** all-NaN column, high-cardinality categorical column, single-class target, mixed types in one column, tiny datasets, very large files
- **Unit tests:** each cleaner function, the problem detector, and the pipeline round trip (save → load → predict)

---

## Getting started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Web Application
python server.py
# Open http://127.0.0.1:5000 in your browser

# 3. Run the automated test suite
pytest tests/
```

---

## Out of scope (for now)

Deployment (FastAPI, Docker), model versioning, monitoring, data drift detection, and automatic retraining.

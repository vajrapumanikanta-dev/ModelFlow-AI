"""
Module 7: Preprocessing Pipeline (core/preprocess.py)
Constructs scikit-learn preprocessing pipelines and enforces the strict rule:
ALWAYS split data BEFORE fitting imputers, scalers, or encoders to prevent data leakage.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    num_impute_strategy: str = "median",
    num_scaler: str = "standard",
    cat_impute_strategy: str = "most_frequent",
    cat_encoder: str = "onehot",
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for numerical and categorical features.
    
    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.
        num_impute_strategy: 'mean', 'median', or 'constant' (0).
        num_scaler: 'standard', 'minmax', 'robust', or 'none'.
        cat_impute_strategy: 'most_frequent' or 'constant' ('Missing').
        cat_encoder: 'onehot' or 'ordinal'.
    """
    transformers = []

    # 1. Numeric Pipeline
    if numeric_cols:
        num_steps = []
        if num_impute_strategy in ["mean", "median"]:
            num_steps.append(("imputer", SimpleImputer(strategy=num_impute_strategy)))
        elif num_impute_strategy == "constant":
            num_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))

        if num_scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif num_scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif num_scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))

        num_pipe = Pipeline(steps=num_steps) if num_steps else "passthrough"
        transformers.append(("num", num_pipe, numeric_cols))

    # 2. Categorical Pipeline
    if categorical_cols:
        cat_steps = []
        if cat_impute_strategy == "most_frequent":
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        elif cat_impute_strategy == "constant":
            cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="Missing")))

        if cat_encoder == "onehot":
            cat_steps.append(
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            )
        elif cat_encoder == "ordinal":
            cat_steps.append(
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                )
            )

        cat_pipe = Pipeline(steps=cat_steps) if cat_steps else "passthrough"
        transformers.append(("cat", cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Extracts output feature names after ColumnTransformer transformation."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual derivation
        names = []
        if hasattr(preprocessor, "named_transformers_"):
            if "num" in preprocessor.named_transformers_ and preprocessor.named_transformers_["num"] != "passthrough":
                names.extend(numeric_cols)
            if "cat" in preprocessor.named_transformers_ and preprocessor.named_transformers_["cat"] != "passthrough":
                cat_step = preprocessor.named_transformers_["cat"]
                if hasattr(cat_step, "named_steps") and "encoder" in cat_step.named_steps:
                    enc = cat_step.named_steps["encoder"]
                    if hasattr(enc, "get_feature_names_out"):
                        names.extend(list(enc.get_feature_names_out(categorical_cols)))
                    else:
                        names.extend(categorical_cols)
                else:
                    names.extend(categorical_cols)
        return names if names else numeric_cols + categorical_cols


def prepare_train_test_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    problem_type: str = "classification",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    num_impute_strategy: str = "median",
    num_scaler: str = "standard",
    cat_impute_strategy: str = "most_frequent",
    cat_encoder: str = "onehot",
    resampling_strategy: str = "none",
) -> Dict[str, Any]:
    """
    Executes proper split-first preprocessing to avoid data leakage:
    1. Separate X and y
    2. Train/Test Split (before any fitting)
    3. Build Preprocessor and fit ONLY on X_train
    4. Transform X_train and X_test
    5. Optionally apply SMOTE / Oversampling / Undersampling strictly on training fold
    """
    if not feature_cols:
        raise ValueError("At least one feature column must be selected.")

    X = df[feature_cols].copy()
    
    # Identify numeric and categorical feature subsets
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Preprocessor definition
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        num_impute_strategy=num_impute_strategy,
        num_scaler=num_scaler,
        cat_impute_strategy=cat_impute_strategy,
        cat_encoder=cat_encoder,
    )

    # 1. Unsupervised branch
    if problem_type == "unsupervised" or not target_col:
        preprocessor.fit(X)
        X_proc = preprocessor.transform(X)
        feature_names_out = get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols)
        return {
            "X_proc": X_proc,
            "X_raw": X,
            "preprocessor": preprocessor,
            "feature_names_out": feature_names_out,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "problem_type": "unsupervised",
        }

    # 2. Supervised branch
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].copy()

    # Drop samples where target is NaN
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # Target label mapping for classification
    target_mapping = {}
    class_names = []
    if problem_type in ["binary_classification", "multiclass_classification"]:
        unique_classes = sorted(y.unique(), key=lambda x: str(x))
        class_names = [str(c) for c in unique_classes]
        target_mapping = {val: idx for idx, val in enumerate(unique_classes)}
        # Map target to integer classes
        y = y.map(target_mapping)
        if y.isna().any():
            # If mapping failed due to types, use factorize
            y_codes, uniques = pd.factorize(df[target_col][valid_mask])
            y = pd.Series(y_codes)
            class_names = [str(u) for u in uniques]

    # Determine stratification
    stratify_labels = None
    if stratify and problem_type in ["binary_classification", "multiclass_classification"]:
        # Verify all classes have at least 2 samples for stratification
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            stratify_labels = y

    # STRICT STEP 1: SPLIT BEFORE FIT
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    # STRICT STEP 2: FIT ONLY ON TRAINING SET
    preprocessor.fit(X_train_raw)

    # STRICT STEP 3: TRANSFORM TRAIN AND TEST INDEPENDENTLY
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    # Get generated column names
    feature_names_out = get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols)

    # STRICT STEP 4: RESAMPLING (ONLY ON X_TRAIN)
    resampling_applied = False
    if problem_type in ["binary_classification", "multiclass_classification"] and resampling_strategy != "none":
        if HAS_IMBLEARN:
            try:
                if resampling_strategy == "smote":
                    # Check that min samples per class in train is > k_neighbors (default 5)
                    min_samples = pd.Series(y_train).value_counts().min()
                    k_neighbors = min(5, max(1, min_samples - 1))
                    if min_samples > 1:
                        sampler = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
                        X_train_proc, y_train = sampler.fit_resample(X_train_proc, y_train)
                        resampling_applied = True
                elif resampling_strategy == "oversample":
                    sampler = RandomOverSampler(random_state=random_state)
                    X_train_proc, y_train = sampler.fit_resample(X_train_proc, y_train)
                    resampling_applied = True
                elif resampling_strategy == "undersample":
                    sampler = RandomUnderSampler(random_state=random_state)
                    X_train_proc, y_train = sampler.fit_resample(X_train_proc, y_train)
                    resampling_applied = True
            except Exception:
                resampling_applied = False

    return {
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
        "y_train": np.array(y_train),
        "y_test": np.array(y_test),
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "preprocessor": preprocessor,
        "feature_names_out": feature_names_out,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "target_mapping": target_mapping,
        "class_names": class_names,
        "problem_type": problem_type,
        "resampling_applied": resampling_applied,
    }

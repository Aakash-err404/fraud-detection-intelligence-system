"""Training module for fraud detection models.

Supports Logistic Regression, Random Forest, and XGBoost.
Handles class imbalance via SMOTE and class weights.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from model.preprocessing import (
    build_preprocessor,
    detect_target_column,
    drop_non_feature_columns,
    normalize_column_names,
    preprocess_data,
    validate_dataframe,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "saved_model.pkl")

AVAILABLE_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
}


def get_model(model_name: str, class_weight_dict: dict | None = None):
    """Instantiate a model by name with sensible defaults."""
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if class_weight_dict is not None else None,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced" if class_weight_dict is not None else None,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        scale = 1.0
        if class_weight_dict and 0 in class_weight_dict and 1 in class_weight_dict:
            scale = class_weight_dict[1] / class_weight_dict[0]
            scale = max(scale, 1.0)
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_class_weights(y: pd.Series) -> dict:
    """Compute class weights from the target distribution."""
    counts = y.value_counts()
    total = len(y)
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return weights


def train_model(
    df: pd.DataFrame,
    model_name: str = "Random Forest",
    test_size: float = 0.2,
    use_smote: bool = True,
    target_col: str | None = None,
) -> dict:
    """Train a fraud detection model on the provided dataframe.

    Returns a dict with:
        - pipeline: trained imblearn Pipeline (preprocessor + SMOTE + model)
        - metrics: evaluation metrics dict
        - confusion_matrix: confusion matrix array
        - model_name: name of the model used
        - target_col: target column name
        - numeric_cols: list of numeric feature columns
        - categorical_cols: list of categorical feature columns
    """
    valid, msg = validate_dataframe(df)
    if not valid:
        raise ValueError(msg)

    # Normalize column names before anything else
    df = normalize_column_names(df)
    df = drop_non_feature_columns(df)

    if target_col is None:
        target_col = detect_target_column(df)
    else:
        # Normalize the user-provided target_col name to match
        target_col = re.sub(r"\s+", "_", target_col.strip().lower())
    if target_col is None:
        raise ValueError(
            "Could not detect target column. Please ensure your dataset has a "
            "column named 'Class', 'is_fraud', 'fraud', 'label', or 'target'."
        )

    X, y, numeric_cols, categorical_cols = preprocess_data(df, target_col)

    if y is None:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    class_weights = None if use_smote else compute_class_weights(y_train)
    classifier = get_model(model_name, class_weights)

    steps = [("preprocessor", preprocessor)]
    if use_smote:
        smote = SMOTE(random_state=42)
        steps.append(("smote", smote))
    steps.append(("classifier", classifier))

    pipeline = ImbPipeline(steps)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "model_name": model_name,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "y_proba": y_proba,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def save_model(result: dict, path: str = DEFAULT_MODEL_PATH) -> str:
    """Save the trained model artifact (pipeline + metadata) to disk."""
    artifact = {
        "pipeline": result["pipeline"],
        "model_name": result["model_name"],
        "target_col": result["target_col"],
        "numeric_cols": result["numeric_cols"],
        "categorical_cols": result["categorical_cols"],
        "metrics": result["metrics"],
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    return path


def load_model(path_or_file=DEFAULT_MODEL_PATH) -> dict:
    """Load a saved model artifact from a file path or file-like object."""
    if hasattr(path_or_file, "read"):
        artifact = pickle.load(path_or_file)
    else:
        with open(path_or_file, "rb") as f:
            artifact = pickle.load(f)
    return artifact

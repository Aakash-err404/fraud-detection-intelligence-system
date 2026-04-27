"""Prediction module for fraud detection.

Loads a pre-trained model and runs inference on new data.
Adapts any incoming dataset to match the model's expected feature columns.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.preprocessing import (
    drop_non_feature_columns,
    normalize_column_names,
    validate_dataframe,
)
from model.train import load_model

FRAUD_THRESHOLD = 0.5


def align_columns(
    df: pd.DataFrame,
    expected_numeric: list[str],
    expected_categorical: list[str],
    target_col: str | None = None,
) -> pd.DataFrame:
    """Align incoming dataframe columns to match expected model features.

    - Normalizes column names (lowercase, trimmed, underscores)
    - Drops obvious non-feature columns (IDs, names, identifiers)
    - Adds missing expected columns with default value 0
    - Drops extra columns not used by the model
    - Reorders columns to match the expected order
    """
    df = normalize_column_names(df)
    df = drop_non_feature_columns(df)

    # Remove the target column if present so it doesn't interfere
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Build mapping: normalized expected name → original expected name
    all_expected = expected_numeric + expected_categorical
    norm_to_orig = {}
    for col in all_expected:
        normalized = re.sub(r"\s+", "_", col.strip().lower())
        norm_to_orig[normalized] = col

    # Rename matched input columns from normalized names to original expected names
    rename_map = {}
    for col in df.columns:
        if col in norm_to_orig:
            rename_map[col] = norm_to_orig[col]
    df = df.rename(columns=rename_map)

    # Add missing expected columns with default value 0
    for col in all_expected:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected columns, in expected order
    df = df[all_expected]
    return df


def predict(
    df: pd.DataFrame,
    model_artifact: dict | None = None,
    model_path: str | None = None,
    threshold: float = FRAUD_THRESHOLD,
) -> pd.DataFrame:
    """Run fraud prediction on a dataframe.

    Automatically adapts the dataset to match the model's expected schema.
    Never raises errors for column mismatches — always aligns instead.

    Args:
        df: Input dataframe with transaction features.
        model_artifact: Pre-loaded model artifact dict. If None, loads from model_path.
        model_path: Path to saved model file. Used only if model_artifact is None.
        threshold: Probability threshold for fraud classification.

    Returns:
        DataFrame with original data plus prediction columns.
    """
    valid, msg = validate_dataframe(df)
    if not valid:
        raise ValueError(msg)

    if model_artifact is None:
        if model_path is None:
            from model.train import DEFAULT_MODEL_PATH
            model_path = DEFAULT_MODEL_PATH
        model_artifact = load_model(model_path)

    pipeline = model_artifact["pipeline"]
    numeric_cols = model_artifact["numeric_cols"]
    categorical_cols = model_artifact["categorical_cols"]
    target_col = model_artifact.get("target_col")

    # Normalize the target_col name to match normalized columns
    normalized_target = re.sub(r"\s+", "_", target_col.strip().lower()) if target_col else None

    X = align_columns(df, numeric_cols, categorical_cols, target_col=normalized_target)

    result_df = df.copy()

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[:, 1]
        result_df["Fraud_Probability"] = np.round(probabilities, 4)
        result_df["Prediction"] = (probabilities >= threshold).astype(int)
    else:
        result_df["Fraud_Probability"] = np.nan
        result_df["Prediction"] = pipeline.predict(X)

    result_df["Prediction_Label"] = result_df["Prediction"].map(
        {0: "Not Fraud", 1: "Fraud"},
    )

    return result_df

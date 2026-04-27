"""Prediction module for fraud detection.

Loads a pre-trained model and runs inference on new data.
If the dataset schema does not match the model features, automatically
retrains on the uploaded dataset instead of forcing column alignment.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from model.preprocessing import (
    detect_target_column,
    normalize_column_names,
    validate_dataframe,
)
from model.train import load_model

FRAUD_THRESHOLD = 0.1


def _columns_match(
    df: pd.DataFrame,
    expected_numeric: list[str],
    expected_categorical: list[str],
) -> bool:
    """Check whether the dataframe columns match the model's expected features.

    Compares using normalized (lowercased) column names so that
    'Amount' matches 'amount', etc.
    """
    df_cols_normalized = {
        re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns
    }
    expected_normalized = {
        re.sub(r"\s+", "_", c.strip().lower())
        for c in expected_numeric + expected_categorical
    }
    matched = expected_normalized & df_cols_normalized
    return len(matched) >= len(expected_normalized) * 0.5


def _prepare_features(
    df: pd.DataFrame,
    expected_numeric: list[str],
    expected_categorical: list[str],
    target_col: str | None = None,
) -> pd.DataFrame:
    """Prepare a feature dataframe by renaming columns to match model expectations.

    Only renames columns that exist in the data; does NOT add missing columns
    with default values (which would produce garbage predictions).
    """
    work = df.copy()

    # Build mapping: normalized name → original expected name
    norm_to_expected: dict[str, str] = {}
    for col in expected_numeric + expected_categorical:
        normalized = re.sub(r"\s+", "_", col.strip().lower())
        norm_to_expected[normalized] = col

    # Normalize data column names and map back to expected names
    rename_map: dict[str, str] = {}
    for col in work.columns:
        normalized = re.sub(r"\s+", "_", col.strip().lower())
        if normalized in norm_to_expected:
            rename_map[col] = norm_to_expected[normalized]
    work = work.rename(columns=rename_map)

    # Drop target column if present
    if target_col:
        norm_target = re.sub(r"\s+", "_", target_col.strip().lower())
        cols_to_drop = [
            c for c in work.columns
            if re.sub(r"\s+", "_", c.strip().lower()) == norm_target
        ]
        if cols_to_drop:
            work = work.drop(columns=cols_to_drop)

    # Keep only the expected columns that actually exist
    all_expected = expected_numeric + expected_categorical
    available = [c for c in all_expected if c in work.columns]
    return work[available]


def predict(
    df: pd.DataFrame,
    model_artifact: dict | None = None,
    model_path: str | None = None,
    threshold: float = FRAUD_THRESHOLD,
) -> pd.DataFrame:
    """Run fraud prediction on a dataframe.

    If the dataset schema matches the model features, runs inference directly.
    If the schema does not match but the dataset contains a target column,
    automatically retrains a model on the uploaded data.

    Args:
        df: Input dataframe with transaction features.
        model_artifact: Pre-loaded model artifact dict.
        model_path: Path to saved model file.
        threshold: Probability threshold for fraud classification (default 0.1).

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

    schema_ok = _columns_match(df, numeric_cols, categorical_cols)

    if not schema_ok:
        # Schema mismatch — check if dataset has a target column for retraining
        detected_target = detect_target_column(
            normalize_column_names(df),
        )
        if detected_target is not None:
            from model.train import train_model
            retrained = train_model(df, model_name="XGBoost")
            pipeline = retrained["pipeline"]
            numeric_cols = retrained["numeric_cols"]
            categorical_cols = retrained["categorical_cols"]
            target_col = retrained["target_col"]
        else:
            raise ValueError(
                "Dataset schema does not match the model's expected features "
                "and no target column was found for automatic retraining. "
                f"Expected features: {numeric_cols + categorical_cols}"
            )

    X = _prepare_features(df, numeric_cols, categorical_cols, target_col)

    if X.shape[1] == 0:
        raise ValueError(
            "No matching feature columns found after preparation. "
            f"Expected: {numeric_cols + categorical_cols}, "
            f"Got: {list(df.columns)}"
        )

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

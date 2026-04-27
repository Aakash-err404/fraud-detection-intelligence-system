"""Preprocessing module for fraud detection pipeline.

Handles missing values, encoding categorical variables, and feature scaling.
Uses sklearn pipelines to ensure consistent preprocessing in training and inference.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Patterns for columns that are obviously not features
_NON_FEATURE_PATTERNS = re.compile(
    r"^(" +
    r"(.*_)?(id|identifier|index|row_number|serial)" +
    r"|transaction_id|trans_id|txn_id|record_id|customer_id|account_id" +
    r"|name|customer_name|cardholder_name|merchant_name|first_name|last_name" +
    r"|full_name|card_number|cc_num|ssn|email|phone|address" +
    r")$",
    re.IGNORECASE,
)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, trim whitespace, replace spaces with underscores."""
    df = df.copy()
    df.columns = [
        re.sub(r"\s+", "_", col.strip().lower())
        for col in df.columns
    ]
    return df


def detect_non_feature_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that are obviously not features (IDs, names, identifiers)."""
    non_feature = []
    for col in df.columns:
        if _NON_FEATURE_PATTERNS.match(col):
            non_feature.append(col)
    return non_feature


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop obvious non-feature columns from the dataframe."""
    to_drop = detect_non_feature_columns(df)
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that the uploaded dataframe has transaction-related features."""
    if df.empty:
        return False, "Uploaded dataset is empty."

    if len(df.columns) < 2:
        return False, "Dataset must contain at least 2 columns."

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return False, "Dataset must contain at least one numeric column."

    return True, "Dataset is valid."


def detect_target_column(df: pd.DataFrame) -> str | None:
    """Auto-detect the target column (fraud label) from the dataframe."""
    target_candidates = [
        "class", "is_fraud", "fraud", "isfraud", "label", "target",
        "isFraud", "Class", "Fraud", "Label", "Target", "IS_FRAUD",
    ]
    for col in target_candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if col.lower() in [c.lower() for c in target_candidates]:
            return col

    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            return col

    return None


def identify_column_types(
    df: pd.DataFrame, target_col: str | None = None,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns, excluding the target."""
    exclude = set()
    if target_col:
        exclude.add(target_col)

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude
    ]
    categorical_cols = [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if col not in exclude
    ]

    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for preprocessing."""
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


def preprocess_data(
    df: pd.DataFrame, target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, list[str], list[str]]:
    """Separate features and target, identify column types.

    Returns:
        X: Feature dataframe
        y: Target series (None if target_col not found)
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
    """
    numeric_cols, categorical_cols = identify_column_types(df, target_col)

    exclude = {target_col} if target_col else set()
    feature_cols = [c for c in numeric_cols + categorical_cols if c not in exclude]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int) if target_col and target_col in df.columns else None

    return X, y, numeric_cols, categorical_cols

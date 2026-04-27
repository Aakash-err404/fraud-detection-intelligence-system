"""Prediction module for fraud detection.

Loads a pre-trained model and runs inference on new data.
"""

import numpy as np
import pandas as pd

from model.preprocessing import validate_dataframe
from model.train import load_model

FRAUD_THRESHOLD = 0.5


def predict(
    df: pd.DataFrame,
    model_artifact: dict | None = None,
    model_path: str | None = None,
    threshold: float = FRAUD_THRESHOLD,
) -> pd.DataFrame:
    """Run fraud prediction on a dataframe.

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

    available_numeric = [c for c in numeric_cols if c in df.columns]
    available_categorical = [c for c in categorical_cols if c in df.columns]

    if not available_numeric and not available_categorical:
        raise ValueError(
            "No matching feature columns found in the uploaded dataset. "
            f"Expected columns: {numeric_cols + categorical_cols}"
        )

    missing_cols = (
        [c for c in numeric_cols if c not in df.columns]
        + [c for c in categorical_cols if c not in df.columns]
    )
    if missing_cols:
        raise ValueError(
            f"The following expected feature columns are missing from the dataset: "
            f"{missing_cols}"
        )

    all_features = numeric_cols + categorical_cols
    X = df[all_features]

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

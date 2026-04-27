"""Generate a pre-trained model using the Credit Card Fraud dataset.

This script downloads the dataset from Kaggle (if available) or generates
a synthetic fraud dataset, trains an XGBoost model, and saves it as
model/saved_model.pkl.

Usage:
    python -m scripts.generate_pretrained_model
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.train import save_model, train_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "saved_model.pkl")


def try_download_kaggle_dataset() -> pd.DataFrame | None:
    """Attempt to download the Credit Card Fraud dataset from Kaggle."""
    try:
        import kaggle  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        api.dataset_download_files(
            "mlg-ulb/creditcardfraud", path=data_dir, unzip=True,
        )
        csv_path = os.path.join(data_dir, "creditcard.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    except Exception:
        pass
    return None


def generate_synthetic_fraud_dataset(
    n_samples: int = 20000, fraud_ratio: float = 0.02,
) -> pd.DataFrame:
    """Generate a synthetic dataset mimicking credit card fraud data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=18,
        n_redundant=5,
        n_classes=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.005,
        random_state=42,
        class_sep=1.5,
    )

    feature_names = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(X, columns=feature_names)

    rng = np.random.RandomState(42)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n_samples)))
    df["Amount"] = np.abs(rng.lognormal(3, 2, n_samples))
    df["Class"] = y

    return df


def main():
    print("Attempting to download Kaggle Credit Card Fraud dataset…")
    df = try_download_kaggle_dataset()

    if df is None:
        print("Kaggle dataset not available. Generating synthetic dataset…")
        df = generate_synthetic_fraud_dataset(n_samples=20000, fraud_ratio=0.02)

    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

    print("\nTraining XGBoost model…")
    result = train_model(
        df,
        model_name="XGBoost",
        test_size=0.2,
        use_smote=True,
        target_col="Class",
    )

    print("\nMetrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}")

    path = save_model(result, MODEL_PATH)
    print(f"\nModel saved to: {path}")
    print(f"File size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

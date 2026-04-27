"""Visualization utilities for the fraud detection system."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    """Create a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: list[str], importances: np.ndarray, top_n: int = 15,
) -> plt.Figure:
    """Plot top N feature importances."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        range(len(indices)),
        importances[indices],
        align="center",
        color="steelblue",
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(indices))} Feature Importances")
    plt.tight_layout()
    return fig


def plot_prediction_distribution(predictions: np.ndarray) -> plt.Figure:
    """Plot distribution of fraud vs not fraud predictions."""
    fig, ax = plt.subplots(figsize=(6, 4))
    unique, counts = np.unique(predictions, return_counts=True)
    labels = ["Not Fraud" if u == 0 else "Fraud" for u in unique]
    colors = ["#2ecc71" if u == 0 else "#e74c3c" for u in unique]
    ax.bar(labels, counts, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(i, count + max(counts) * 0.01, str(count), ha="center", va="bottom")
    plt.tight_layout()
    return fig


def plot_probability_distribution(probabilities: np.ndarray) -> plt.Figure:
    """Plot distribution of fraud probabilities."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probabilities, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Fraud Probabilities")
    ax.legend()
    plt.tight_layout()
    return fig

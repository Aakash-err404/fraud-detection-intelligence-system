"""Visualization utilities for the fraud detection system."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# ── Analytics visualizations ────────────────────────────────────────────────


def plot_anomaly_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> plt.Figure:
    """Scatter plot coloured by anomaly label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = df[df["Anomaly"] == 1]
    anomaly = df[df["Anomaly"] == -1]
    ax.scatter(
        normal[x_col], normal[y_col],
        c="#2ecc71", alpha=0.5, s=20, label="Normal",
    )
    ax.scatter(
        anomaly[x_col], anomaly[y_col],
        c="#e74c3c", alpha=0.8, s=40, marker="x", label="Anomaly",
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Anomaly Detection Results")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_anomaly_score_distribution(scores: np.ndarray) -> plt.Figure:
    """Histogram of Isolation-Forest anomaly scores."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", label="Decision boundary")
    ax.set_xlabel("Anomaly Score (lower = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Anomaly Scores")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_cluster_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> plt.Figure:
    """Scatter plot coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("husl", df["Cluster"].nunique())
    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        label = subset["Cluster_Label"].iloc[0] if "Cluster_Label" in subset else str(cluster_id)
        ax.scatter(
            subset[x_col], subset[y_col],
            color=palette[cluster_id], alpha=0.6, s=30, label=label,
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("User Segmentation (K-Means Clustering)")
    ax.legend(title="Segment")
    plt.tight_layout()
    return fig


def plot_cluster_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of cluster sizes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Cluster_Label"].value_counts().sort_index()
    colors = sns.color_palette("husl", len(counts))
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Cluster Distribution")
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, count + max(counts.values) * 0.01,
            str(count), ha="center", va="bottom",
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_spending_score_distribution(scores: pd.Series) -> plt.Figure:
    """Histogram of spending scores with segment bands."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    q33 = scores.quantile(0.33)
    q66 = scores.quantile(0.66)
    ax.axvline(x=q33, color="orange", linestyle="--", label=f"33rd pctl ({q33:.3f})")
    ax.axvline(x=q66, color="red", linestyle="--", label=f"66th pctl ({q66:.3f})")
    ax.set_xlabel("Spending Score")
    ax.set_ylabel("Count")
    ax.set_title("Spending Score Distribution")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_spending_segments(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of spending-score segments."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Spending_Segment"].value_counts()
    order = ["Normal User", "Active User", "High Value / Risky"]
    counts = counts.reindex([o for o in order if o in counts.index])
    color_map = {"Normal User": "#2ecc71", "Active User": "#f39c12", "High Value / Risky": "#e74c3c"}
    colors = [color_map.get(seg, "#95a5a6") for seg in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Spending Segments")
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, count + max(counts.values) * 0.01,
            str(count), ha="center", va="bottom",
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig

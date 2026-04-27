"""Analytics module — data-mining techniques from the case study.

Provides:
  1. Anomaly Detection   (Isolation Forest)
  2. User Clustering     (K-Means segmentation)
  3. Spending Score      (weighted regression formula)
  4. Rule-based Fraud    (heuristic fraud indicators)
  5. Association Rules   (Apriori via mlxtend, optional)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Isolation-Forest anomaly detection on *df*.

    Returns a copy of *df* with two extra columns:
      - ``Anomaly``       : -1 (anomaly) or 1 (normal)
      - ``Anomaly_Score``  : continuous score (lower → more anomalous)
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not feature_cols:
        raise ValueError("No numeric columns available for anomaly detection.")

    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    result = df.copy()
    result["Anomaly"] = labels
    result["Anomaly_Score"] = scores
    result["Anomaly_Label"] = np.where(labels == -1, "Anomaly", "Normal")
    return result


# ---------------------------------------------------------------------------
# 2. User Clustering (K-Means)
# ---------------------------------------------------------------------------

def cluster_users(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Segment users/transactions into *n_clusters* groups via K-Means.

    Returns:
      - DataFrame with ``Cluster`` and ``Cluster_Label`` columns
      - fitted KMeans estimator
      - fitted StandardScaler
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not feature_cols:
        raise ValueError("No numeric columns available for clustering.")

    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    result = df.copy()
    result["Cluster"] = labels

    cluster_means = result.groupby("Cluster")[feature_cols].mean()
    amount_col = _find_amount_col(feature_cols)
    if amount_col:
        order = cluster_means[amount_col].sort_values().index.tolist()
    else:
        order = cluster_means.mean(axis=1).sort_values().index.tolist()

    if n_clusters == 2:
        segment_names = ["Low Spenders", "High Spenders"]
    elif n_clusters == 3:
        segment_names = ["Low Spenders", "Regular Users", "High Spenders"]
    else:
        segment_names = [f"Segment {i}" for i in range(n_clusters)]
    label_map = {old: segment_names[i] for i, old in enumerate(order)}
    result["Cluster_Label"] = result["Cluster"].map(label_map)

    return result, km, scaler


def _find_amount_col(cols: list[str]) -> str | None:
    """Heuristic to find the transaction-amount column."""
    for c in cols:
        if c.lower() in ("amount", "transaction_amount", "amt", "transactionamt"):
            return c
    return None


# ---------------------------------------------------------------------------
# 3. Spending Score (Regression)
# ---------------------------------------------------------------------------

def compute_spending_score(
    df: pd.DataFrame,
    amount_col: str | None = None,
    frequency_col: str | None = None,
    time_col: str | None = None,
    category_col: str | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute a Spending Score per the case-study formula.

    Formula (default weights):
        Score = 0.3 * Amount + 0.2 * Frequency + 0.3 * Time + 0.2 * Category

    All features are min-max scaled to [0, 1] before weighting so the score
    stays interpretable regardless of original magnitudes.

    Returns *df* with ``Spending_Score`` and ``Spending_Segment`` columns.
    """
    if weights is None:
        weights = {"amount": 0.3, "frequency": 0.2, "time": 0.3, "category": 0.2}

    result = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    all_cols = df.columns.tolist()

    def _resolve(col_hint: str | None, keywords: list[str], search_cols: list[str] | None = None) -> str | None:
        if col_hint and col_hint in df.columns:
            return col_hint
        for c in (search_cols if search_cols is not None else numeric_cols):
            if c.lower() in keywords:
                return c
        return None

    amt = _resolve(amount_col, ["amount", "amt", "transaction_amount", "transactionamt"])
    freq = _resolve(frequency_col, ["frequency", "freq", "transaction_frequency"])
    time = _resolve(time_col, ["time", "hour", "timestamp"])
    cat = _resolve(category_col, ["category", "merchant_category", "merchantcategory"], search_cols=all_cols)

    def _scale_col(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    score = pd.Series(0.0, index=df.index)
    components_used = 0

    if amt:
        score += weights["amount"] * _scale_col(df[amt].fillna(0))
        components_used += 1
    if freq:
        score += weights["frequency"] * _scale_col(df[freq].fillna(0))
        components_used += 1
    if time:
        score += weights["time"] * _scale_col(df[time].fillna(0))
        components_used += 1
    if cat:
        encoded = df[cat].astype("category").cat.codes.astype(float)
        score += weights["category"] * _scale_col(encoded)
        components_used += 1

    if components_used == 0:
        for c in numeric_cols[:4]:
            score += (1 / min(len(numeric_cols), 4)) * _scale_col(df[c].fillna(0))
        components_used = min(len(numeric_cols), 4)

    result["Spending_Score"] = score

    q33 = score.quantile(0.33)
    q66 = score.quantile(0.66)
    if q33 == q66:
        result["Spending_Segment"] = pd.Series(
            "Active User", index=score.index, dtype="category",
        )
    else:
        result["Spending_Segment"] = pd.cut(
            score,
            bins=[-np.inf, q33, q66, np.inf],
            labels=["Normal User", "Active User", "High Value / Risky"],
        )
    return result


# ---------------------------------------------------------------------------
# 4. Rule-based Fraud Indicators
# ---------------------------------------------------------------------------

def apply_fraud_rules(
    df: pd.DataFrame,
    amount_threshold: float = 50_000,
    late_night_hours: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Apply heuristic rules from the case study and flag suspicious rows.

    Rules:
      R1: Amount > threshold AND late night  → Fraud flag
      R2: New location + High amount         → Fraud flag  (location col optional)
      R3: High frequency in short time       → Fraud flag  (frequency col optional)

    Returns *df* with ``Rule_Flag`` (bool) and ``Rule_Reasons`` (str) columns.
    """
    orig_index = df.index
    work = df.reset_index(drop=True)
    flags = pd.Series(False, index=work.index)
    reasons: list[list[str]] = [[] for _ in range(len(work))]

    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()

    amt_col = _find_amount_col(numeric_cols)
    hour_col = None
    for c in numeric_cols:
        if c.lower() in ("hour", "time"):
            if work[c].max() <= 23:
                hour_col = c
                break

    freq_col = None
    for c in numeric_cols:
        if c.lower() in ("frequency", "freq", "transaction_frequency"):
            freq_col = c
            break

    location_col = None
    for c in work.columns:
        if c.lower() in ("location", "loc", "city", "region"):
            location_col = c
            break

    # Rule 1: High amount + late night
    if amt_col and hour_col:
        mask = (work[amt_col] > amount_threshold) & (work[hour_col].isin(late_night_hours))
        flags |= mask
        for i in mask[mask].index:
            reasons[i].append("High amount + late night")
    elif amt_col:
        mask = work[amt_col] > amount_threshold
        flags |= mask
        for i in mask[mask].index:
            reasons[i].append(f"Amount > {amount_threshold:,.0f}")

    # Rule 2: New / unusual location + high amount
    if location_col and amt_col:
        loc_counts = work[location_col].value_counts()
        rare_locs = loc_counts[loc_counts <= 2].index
        mask = work[location_col].isin(rare_locs) & (work[amt_col] > amount_threshold * 0.5)
        flags |= mask
        for i in mask[mask].index:
            reasons[i].append("Rare location + elevated amount")

    # Rule 3: High frequency in short time
    if freq_col:
        high_freq = work[freq_col] > work[freq_col].quantile(0.95)
        flags |= high_freq
        for i in high_freq[high_freq].index:
            reasons[i].append("High transaction frequency")

    result = df.copy()
    result["Rule_Flag"] = flags.values
    result["Rule_Reasons"] = ["; ".join(r) if r else "" for r in reasons]
    return result


# ---------------------------------------------------------------------------
# 5. Association Rule Mining (optional — requires mlxtend)
# ---------------------------------------------------------------------------

def mine_association_rules(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    n_bins: int = 3,
) -> pd.DataFrame | None:
    """Discover association rules using the Apriori algorithm.

    Numeric columns are discretized into *n_bins* bins before mining.
    Returns a DataFrame of rules or ``None`` if mlxtend is not installed.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        return None

    if columns is None:
        columns = df.columns.tolist()

    bin_labels = ["Low", "Med", "High"][:n_bins] if n_bins <= 3 else [f"Bin_{i}" for i in range(1, n_bins + 1)]

    # Pre-compute binned values for numeric columns across the full column
    binned_cols: dict[str, pd.Series] = {}
    for col in columns:
        if df[col].dtype.kind in ("i", "f"):
            try:
                binned_cols[col] = pd.cut(
                    df[col], bins=n_bins, labels=bin_labels,
                )
            except Exception:
                binned_cols[col] = pd.Series("Med", index=df.index)

    items_per_row: list[list[str]] = []
    for idx, row in df[columns].iterrows():
        items: list[str] = []
        for col in columns:
            val = row[col]
            if pd.isna(val):
                continue
            if col in binned_cols:
                bl = binned_cols[col].loc[idx]
                if pd.isna(bl):
                    continue
                items.append(f"{col}={bl}")
            else:
                items.append(f"{col}={val}")
        items_per_row.append(items)

    te = TransactionEncoder()
    te_array = te.fit(items_per_row).transform(items_per_row)
    basket = pd.DataFrame(te_array, columns=te.columns_)

    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].sort_values("lift", ascending=False)

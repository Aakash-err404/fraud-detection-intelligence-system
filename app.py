"""Fraud Detection Intelligence System — Streamlit application."""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from model.analytics import (
    apply_fraud_rules,
    cluster_users,
    compute_spending_score,
    detect_anomalies,
    mine_association_rules,
)
from model.predict import predict
from model.preprocessing import detect_target_column, validate_dataframe
from model.train import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_PATH,
    load_model,
    save_model,
    train_model,
)
from utils.visualization import (
    plot_anomaly_scatter,
    plot_anomaly_score_distribution,
    plot_cluster_distribution,
    plot_cluster_scatter,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_prediction_distribution,
    plot_probability_distribution,
    plot_spending_score_distribution,
    plot_spending_segments,
)

st.set_page_config(
    page_title="Fraud Detection Intelligence System",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Fraud Detection Intelligence System")
st.markdown(
    "Upload a transaction dataset and detect fraud using machine learning models."
)

st.sidebar.header("Configuration")
mode = st.sidebar.radio(
    "Select Mode",
    ["Use Pre-trained Model", "Train New Model", "Analytics Dashboard"],
    help="Choose to use an existing trained model, train a new one, or run analytics.",
)


def display_metrics(metrics: dict) -> None:
    """Display evaluation metrics in a 4-column layout."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")


def get_feature_names_from_pipeline(pipeline, numeric_cols, categorical_cols):
    """Extract feature names from a fitted pipeline's preprocessor."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        return None

    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)
        elif name == "cat":
            encoder = transformer.named_steps.get("encoder")
            if encoder is not None and hasattr(encoder, "get_feature_names_out"):
                feature_names.extend(encoder.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
    return feature_names


# ── Pre-trained Model Mode ──────────────────────────────────────────────────

if mode == "Use Pre-trained Model":
    st.header("📊 Predict with Pre-trained Model")

    model_file = st.sidebar.file_uploader(
        "Upload custom model (.pkl)", type=["pkl"],
        help="Optional: upload your own trained model. Otherwise the default is used.",
    )

    model_artifact = None
    if model_file is not None:
        st.sidebar.warning(
            "⚠️ Only load model files from trusted sources. "
            "Malicious .pkl files can execute arbitrary code."
        )
        try:
            model_artifact = load_model(model_file)
            required_keys = {"pipeline", "model_name", "numeric_cols", "categorical_cols"}
            if not isinstance(model_artifact, dict) or not required_keys.issubset(model_artifact):
                st.sidebar.error(
                    "Invalid model file. Expected keys: "
                    + ", ".join(sorted(required_keys))
                )
                model_artifact = None
            else:
                st.sidebar.success("Custom model loaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
    elif os.path.exists(DEFAULT_MODEL_PATH):
        try:
            model_artifact = load_model(DEFAULT_MODEL_PATH)
            st.sidebar.info("Using default pre-trained model.")
        except Exception as e:
            st.sidebar.warning(f"Could not load default model: {e}")

    if model_artifact is None:
        st.warning(
            "No pre-trained model found. Please train a model first or upload one."
        )
    else:
        st.info(
            f"**Model:** {model_artifact.get('model_name', 'Unknown')} · "
            f"**Features:** {len(model_artifact.get('numeric_cols', []))} numeric, "
            f"{len(model_artifact.get('categorical_cols', []))} categorical"
        )
        if "metrics" in model_artifact:
            with st.expander("Training Metrics", expanded=False):
                display_metrics(model_artifact["metrics"])

    uploaded = st.file_uploader("Upload transaction dataset (CSV)", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

            valid, msg = validate_dataframe(df)
            if not valid:
                st.error(msg)
            elif model_artifact is not None:
                if st.button("🚀 Run Prediction", type="primary"):
                    with st.spinner("Running inference…"):
                        result_df = predict(df, model_artifact=model_artifact)

                    # Enrich with rule-based flags & anomaly scores
                    try:
                        result_df = apply_fraud_rules(result_df)
                    except Exception:
                        pass
                    try:
                        numeric_feat = [
                            c for c in result_df.select_dtypes(include=[np.number]).columns
                            if c not in ("Prediction", "Fraud_Probability", "Anomaly", "Anomaly_Score", "Rule_Flag")
                        ]
                        if len(numeric_feat) >= 2:
                            result_df = detect_anomalies(result_df, feature_cols=numeric_feat)
                    except Exception:
                        pass

                    st.subheader("Prediction Results")

                    fraud_count = (result_df["Prediction"] == 1).sum()
                    not_fraud_count = (result_df["Prediction"] == 0).sum()
                    anomaly_count = (result_df["Anomaly"] == -1).sum() if "Anomaly" in result_df.columns else None
                    rule_flag_count = result_df["Rule_Flag"].sum() if "Rule_Flag" in result_df.columns else None

                    cols_m = st.columns(4 if anomaly_count is not None else 2)
                    cols_m[0].metric("Fraud Detected", fraud_count)
                    cols_m[1].metric("Not Fraud", not_fraud_count)
                    if anomaly_count is not None:
                        cols_m[2].metric("Anomalies", anomaly_count)
                    if rule_flag_count is not None and len(cols_m) > 3:
                        cols_m[3].metric("Rule Flags", int(rule_flag_count))

                    display_cols = ["Prediction_Label", "Fraud_Probability"]
                    if "Rule_Flag" in result_df.columns:
                        display_cols.append("Rule_Reasons")
                    if "Anomaly_Label" in result_df.columns:
                        display_cols.append("Anomaly_Label")
                    extra = [c for c in display_cols if c in result_df.columns]
                    st.dataframe(
                        result_df[extra + [c for c in result_df.columns if c not in extra]],
                        use_container_width=True,
                    )

                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.pyplot(
                            plot_prediction_distribution(
                                result_df["Prediction"].values,
                            ),
                        )
                    with col_c2:
                        if "Fraud_Probability" in result_df.columns:
                            probs = result_df["Fraud_Probability"].dropna().values
                            if len(probs) > 0:
                                st.pyplot(plot_probability_distribution(probs))

                    target_col = detect_target_column(result_df)
                    if target_col and target_col in df.columns:
                        from sklearn.metrics import (
                            accuracy_score,
                            confusion_matrix,
                            f1_score,
                            precision_score,
                            recall_score,
                        )

                        y_true = df[target_col].astype(int)
                        y_pred = result_df["Prediction"]
                        eval_metrics = {
                            "accuracy": accuracy_score(y_true, y_pred),
                            "precision": precision_score(y_true, y_pred, zero_division=0),
                            "recall": recall_score(y_true, y_pred, zero_division=0),
                            "f1_score": f1_score(y_true, y_pred, zero_division=0),
                        }
                        st.subheader("Evaluation Metrics")
                        display_metrics(eval_metrics)
                        cm = confusion_matrix(y_true, y_pred)
                        st.pyplot(plot_confusion_matrix(cm))

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions",
                        csv,
                        "predictions.csv",
                        "text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ── Train New Model Mode ────────────────────────────────────────────────────

elif mode == "Train New Model":
    st.header("🏋️ Train a New Model")

    model_choice = st.sidebar.selectbox(
        "Select Model", list(AVAILABLE_MODELS.keys()),
    )
    use_smote = st.sidebar.checkbox("Apply SMOTE (class imbalance)", value=True)
    test_size = st.sidebar.slider("Test split ratio", 0.1, 0.4, 0.2, 0.05)

    uploaded_train = st.file_uploader(
        "Upload training dataset (CSV)", type=["csv"],
        help="Dataset must include a target column (Class, is_fraud, fraud, label, target).",
    )

    if uploaded_train is not None:
        try:
            df_train = pd.read_csv(uploaded_train)
            st.subheader("Dataset Preview")
            st.dataframe(df_train.head(20), use_container_width=True)
            st.caption(f"{df_train.shape[0]} rows × {df_train.shape[1]} columns")

            target = detect_target_column(df_train)
            if target:
                st.info(f"Detected target column: **{target}**")
                class_dist = df_train[target].value_counts()
                st.write("Class distribution:", class_dist.to_dict())
            else:
                st.warning(
                    "Could not auto-detect target column. Training may fail."
                )

            valid, msg = validate_dataframe(df_train)
            if not valid:
                st.error(msg)
            elif st.button("🏋️ Train Model", type="primary"):
                with st.spinner(f"Training {model_choice}…"):
                    result = train_model(
                        df_train,
                        model_name=model_choice,
                        test_size=test_size,
                        use_smote=use_smote,
                        target_col=target,
                    )

                st.success(f"{model_choice} trained successfully!")

                st.subheader("Evaluation Metrics")
                display_metrics(result["metrics"])

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_confusion_matrix(result["confusion_matrix"]))
                with col2:
                    st.pyplot(
                        plot_prediction_distribution(result["y_pred"]),
                    )

                if result.get("y_proba") is not None:
                    st.pyplot(
                        plot_probability_distribution(result["y_proba"]),
                    )

                pipeline = result["pipeline"]
                classifier = pipeline.named_steps.get("classifier")
                if hasattr(classifier, "feature_importances_"):
                    feature_names = get_feature_names_from_pipeline(
                        pipeline,
                        result["numeric_cols"],
                        result["categorical_cols"],
                    )
                    if feature_names and len(feature_names) == len(
                        classifier.feature_importances_,
                    ):
                        st.subheader("Feature Importance")
                        st.pyplot(
                            plot_feature_importance(
                                feature_names,
                                classifier.feature_importances_,
                            ),
                        )

                with st.expander("Classification Report"):
                    st.text(result["classification_report"])

                saved_path = save_model(result)
                st.info(f"Model saved to `{saved_path}`")

        except Exception as e:
            st.error(f"Error: {e}")

# ── Analytics Dashboard Mode ────────────────────────────────────────────────

else:
    st.header("📊 Analytics Dashboard")
    st.markdown(
        "Run advanced data-mining analyses: anomaly detection, user clustering, "
        "spending-score regression, and association-rule mining."
    )

    uploaded_analytics = st.file_uploader(
        "Upload transaction dataset (CSV)", type=["csv"],
        help="Upload any transaction CSV for analytics.",
    )

    if uploaded_analytics is not None:
        try:
            df_a = pd.read_csv(uploaded_analytics)
            st.subheader("Dataset Preview")
            st.dataframe(df_a.head(20), use_container_width=True)
            st.caption(f"{df_a.shape[0]} rows × {df_a.shape[1]} columns")

            valid, msg = validate_dataframe(df_a)
            if not valid:
                st.error(msg)
            else:
                numeric_cols_a = df_a.select_dtypes(include=[np.number]).columns.tolist()
                target_col_a = detect_target_column(df_a)
                analysis_features = [
                    c for c in numeric_cols_a
                    if target_col_a is None or c != target_col_a
                ]

                tabs = st.tabs([
                    "🔍 Anomaly Detection",
                    "👥 User Clustering",
                    "💰 Spending Score",
                    "🚩 Rule-based Flags",
                    "🔗 Association Rules",
                ])

                # ── Tab 1: Anomaly Detection ─────────────────────
                with tabs[0]:
                    st.subheader("Anomaly Detection (Isolation Forest)")
                    contamination = st.slider(
                        "Contamination (expected anomaly fraction)",
                        0.01, 0.20, 0.05, 0.01,
                    )
                    if st.button("🔍 Detect Anomalies", type="primary"):
                        with st.spinner("Running Isolation Forest…"):
                            df_anom = detect_anomalies(
                                df_a,
                                feature_cols=analysis_features,
                                contamination=contamination,
                            )
                        anom_count = (df_anom["Anomaly"] == -1).sum()
                        normal_count = (df_anom["Anomaly"] == 1).sum()
                        ca, cb = st.columns(2)
                        ca.metric("Anomalies Detected", anom_count)
                        cb.metric("Normal Transactions", normal_count)

                        if len(analysis_features) >= 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plot_anomaly_scatter(
                                    df_anom, analysis_features[0], analysis_features[1],
                                ))
                            with col2:
                                st.pyplot(plot_anomaly_score_distribution(
                                    df_anom["Anomaly_Score"].values,
                                ))

                        st.subheader("Anomalous Transactions")
                        st.dataframe(
                            df_anom[df_anom["Anomaly"] == -1].head(50),
                            use_container_width=True,
                        )
                        csv_anom = df_anom.to_csv(index=False)
                        st.download_button(
                            "📥 Download Anomaly Results", csv_anom,
                            "anomaly_results.csv", "text/csv",
                        )

                # ── Tab 2: User Clustering ───────────────────────
                with tabs[1]:
                    st.subheader("User Segmentation (K-Means Clustering)")
                    n_clusters = st.slider("Number of clusters", 2, 6, 3)
                    if st.button("👥 Run Clustering", type="primary"):
                        with st.spinner("Running K-Means…"):
                            df_clust, km, scaler = cluster_users(
                                df_a,
                                feature_cols=analysis_features,
                                n_clusters=n_clusters,
                            )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plot_cluster_distribution(df_clust))
                        with col2:
                            if len(analysis_features) >= 2:
                                st.pyplot(plot_cluster_scatter(
                                    df_clust,
                                    analysis_features[0],
                                    analysis_features[1],
                                ))

                        st.subheader("Cluster Summary")
                        summary = df_clust.groupby("Cluster_Label")[
                            analysis_features
                        ].mean()
                        st.dataframe(summary, use_container_width=True)

                        csv_clust = df_clust.to_csv(index=False)
                        st.download_button(
                            "📥 Download Clustering Results", csv_clust,
                            "clustering_results.csv", "text/csv",
                        )

                # ── Tab 3: Spending Score ────────────────────────
                with tabs[2]:
                    st.subheader("Spending Score (Regression)")
                    st.markdown(
                        "Computes a weighted spending score per the case-study formula: "
                        "`Score = 0.3 × Amount + 0.2 × Frequency + 0.3 × Time + 0.2 × Category` "
                        "(features are auto-detected and min-max scaled)."
                    )
                    if st.button("💰 Compute Spending Scores", type="primary"):
                        with st.spinner("Computing scores…"):
                            df_score = compute_spending_score(df_a)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plot_spending_score_distribution(
                                df_score["Spending_Score"],
                            ))
                        with col2:
                            st.pyplot(plot_spending_segments(df_score))

                        st.subheader("Segment Summary")
                        seg_summary = df_score.groupby("Spending_Segment")[
                            analysis_features
                        ].mean()
                        st.dataframe(seg_summary, use_container_width=True)

                        csv_score = df_score.to_csv(index=False)
                        st.download_button(
                            "📥 Download Spending Scores", csv_score,
                            "spending_scores.csv", "text/csv",
                        )

                # ── Tab 4: Rule-based Fraud Flags ────────────────
                with tabs[3]:
                    st.subheader("Rule-based Fraud Indicators")
                    st.markdown(
                        "Heuristic rules from the case study:\n"
                        "- **R1**: Amount > threshold AND late night → Fraud\n"
                        "- **R2**: Rare location + elevated amount → Fraud\n"
                        "- **R3**: High transaction frequency → Fraud"
                    )
                    if st.button("🚩 Apply Fraud Rules", type="primary"):
                        with st.spinner("Applying rules…"):
                            df_rules = apply_fraud_rules(df_a)

                        flagged = df_rules["Rule_Flag"].sum()
                        cr, cn = st.columns(2)
                        cr.metric("Flagged Transactions", int(flagged))
                        cn.metric("Clean Transactions", len(df_rules) - int(flagged))

                        if flagged > 0:
                            st.subheader("Flagged Transactions")
                            st.dataframe(
                                df_rules[df_rules["Rule_Flag"]][
                                    ["Rule_Reasons"] + [
                                        c for c in df_rules.columns
                                        if c not in ("Rule_Flag", "Rule_Reasons")
                                    ]
                                ],
                                use_container_width=True,
                            )

                        csv_rules = df_rules.to_csv(index=False)
                        st.download_button(
                            "📥 Download Rule Results", csv_rules,
                            "rule_based_results.csv", "text/csv",
                        )

                # ── Tab 5: Association Rules ─────────────────────
                with tabs[4]:
                    st.subheader("Association Rule Mining")
                    st.markdown(
                        "Discover hidden patterns using the Apriori algorithm. "
                        "Numeric features are discretized into Low / Med / High bins."
                    )
                    ar_cols = st.multiselect(
                        "Select columns for mining",
                        df_a.columns.tolist(),
                        default=df_a.columns.tolist()[:5],
                    )
                    min_support = st.slider("Min support", 0.01, 0.30, 0.05, 0.01)
                    min_confidence = st.slider("Min confidence", 0.1, 1.0, 0.5, 0.05)

                    if st.button("🔗 Mine Rules", type="primary"):
                        with st.spinner("Mining association rules…"):
                            rules = mine_association_rules(
                                df_a,
                                columns=ar_cols,
                                min_support=min_support,
                                min_confidence=min_confidence,
                            )
                        if rules is None:
                            st.warning(
                                "Association-rule mining requires the `mlxtend` library. "
                                "Install it with: `pip install mlxtend`"
                            )
                        elif rules.empty:
                            st.info(
                                "No rules found with the current thresholds. "
                                "Try lowering the minimum support or confidence."
                            )
                        else:
                            st.success(f"Found {len(rules)} association rules.")
                            st.dataframe(rules, use_container_width=True)
                            csv_rules_ar = rules.to_csv(index=False)
                            st.download_button(
                                "📥 Download Rules", csv_rules_ar,
                                "association_rules.csv", "text/csv",
                            )

        except Exception as e:
            st.error(f"Error: {e}")


# ── Footer ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Fraud Detection Intelligence System**  \n"
    "Built with Streamlit, scikit-learn & XGBoost"
)

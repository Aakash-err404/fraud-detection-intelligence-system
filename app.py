"""Fraud Detection Intelligence System — Streamlit application."""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

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
    plot_confusion_matrix,
    plot_feature_importance,
    plot_prediction_distribution,
    plot_probability_distribution,
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
    ["Use Pre-trained Model", "Train New Model"],
    help="Choose to use an existing trained model or train a new one.",
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

                    st.subheader("Prediction Results")

                    fraud_count = (result_df["Prediction"] == 1).sum()
                    not_fraud_count = (result_df["Prediction"] == 0).sum()
                    col_a, col_b = st.columns(2)
                    col_a.metric("Fraud Detected", fraud_count)
                    col_b.metric("Not Fraud", not_fraud_count)

                    display_cols = ["Prediction_Label", "Fraud_Probability"]
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

else:
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


# ── Footer ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Fraud Detection Intelligence System**  \n"
    "Built with Streamlit, scikit-learn & XGBoost"
)

# 🔍 Fraud Detection Intelligence System

A web-based fraud detection system that uses machine learning to classify transactions as **Fraud** or **Not Fraud**. Built with Streamlit, scikit-learn, and XGBoost.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aakash-err404/fraud-detection-intelligence-system/blob/main/notebooks/Fraud_Detection_Intelligence_System.ipynb)

## Features

- **CSV Upload** — Upload any transaction dataset for fraud detection
- **Pre-trained Model** — Instant predictions using a bundled XGBoost model
- **Train New Models** — Train Logistic Regression, Random Forest, or XGBoost on your data
- **Class Imbalance Handling** — SMOTE oversampling and balanced class weights
- **Evaluation Metrics** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Feature Importance** — Visualize which features drive fraud predictions
- **Fraud Probability Scores** — Per-transaction probability alongside binary labels
- **Analytics Dashboard** — Five data-mining techniques from the case study:
  - **Anomaly Detection** — Isolation Forest to flag unusual transactions
  - **User Clustering** — K-Means segmentation (Low Spenders / Regular / High Spenders)
  - **Spending Score** — Weighted regression score with segment classification
  - **Rule-based Fraud Flags** — Heuristic indicators (high amount + late night, rare location, high frequency)
  - **Association Rule Mining** — Apriori algorithm to discover hidden transaction patterns
- **Enriched Predictions** — ML predictions augmented with anomaly scores and rule-based flags
- **Downloadable Results** — Export predictions, anomaly results, and clustering as CSV

## Project Structure

```
fraud_detection_system/
├── app.py                  # Streamlit application
├── model/
│   ├── preprocessing.py    # Data validation, feature detection, sklearn pipelines
│   ├── train.py            # Model training (LR, RF, XGBoost) with SMOTE
│   ├── predict.py          # Inference with threshold-based classification
│   ├── analytics.py        # Anomaly detection, clustering, spending score, rules, association mining
│   └── saved_model.pkl     # Pre-trained XGBoost model
├── utils/
│   └── visualization.py    # Confusion matrix, feature importance, distribution plots
├── scripts/
│   └── generate_pretrained_model.py  # Script to regenerate the pre-trained model
├── notebooks/
│   └── Fraud_Detection_Intelligence_System.ipynb  # Google Colab notebook
├── data/                   # Data directory (datasets go here)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Aakash-err404/fraud-detection-intelligence-system.git
cd fraud-detection-intelligence-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

### Pre-trained Model Mode

1. Select **"Use Pre-trained Model"** in the sidebar
2. Upload a CSV file with transaction features
3. Click **"Run Prediction"**
4. View results: fraud/not-fraud labels, probabilities, metrics, and charts

### Train New Model Mode

1. Select **"Train New Model"** in the sidebar
2. Upload a labeled dataset (must include a target column like `Class`, `is_fraud`, `fraud`, `label`, or `target`)
3. Choose a model: Logistic Regression, Random Forest, or XGBoost
4. Toggle SMOTE and adjust the test split ratio
5. Click **"Train Model"**
6. View evaluation metrics and download the trained model

### Analytics Dashboard Mode

1. Select **"Analytics Dashboard"** in the sidebar
2. Upload any transaction CSV
3. Use the five tabs:
   - **Anomaly Detection** — Adjust contamination rate, run Isolation Forest, view scatter plot and anomalous transactions
   - **User Clustering** — Choose cluster count (2–6), run K-Means, view segment distribution and summary
   - **Spending Score** — Compute weighted spending scores, view distribution and segment breakdown
   - **Rule-based Flags** — Apply heuristic fraud rules, view flagged transactions with reasons
   - **Association Rules** — Select columns, set support/confidence thresholds, mine Apriori rules

### Dataset Requirements

Your CSV should contain:
- **Numeric features** — Transaction amounts, timestamps, PCA components, etc.
- **Target column** (for training) — Binary column named `Class`, `is_fraud`, `fraud`, `label`, or `target` with values `0` (not fraud) and `1` (fraud)

The system automatically handles:
- Missing values (median imputation for numeric, mode for categorical)
- Categorical encoding (one-hot encoding)
- Feature scaling (standardization)

## Google Colab Notebook

The notebook at `notebooks/Fraud_Detection_Intelligence_System.ipynb` provides the full pipeline:
1. Load dataset (Kaggle download, file upload, or synthetic generation)
2. Exploratory data analysis with visualizations
3. Preprocessing with sklearn pipelines
4. Train and compare three models
5. Feature importance analysis
6. Save and demonstrate the best model

## Technical Details

- **Preprocessing** — sklearn `ColumnTransformer` with `Pipeline` ensures identical transforms in training and inference (no data leakage)
- **Class Imbalance** — SMOTE (via `imbalanced-learn`) + balanced class weights
- **Models** — Logistic Regression, Random Forest (200 trees), XGBoost (200 rounds)
- **Threshold** — Default 0.5 (configurable in `model/predict.py`)
- **Serialization** — Full pipeline (preprocessor + SMOTE + classifier) saved as `.pkl`
- **Analytics** — Isolation Forest anomaly detection, K-Means clustering, weighted spending-score regression, heuristic rule-based fraud flags, Apriori association-rule mining (via mlxtend)

## Regenerating the Pre-trained Model

```bash
python -m scripts.generate_pretrained_model
```

This trains an XGBoost model on a synthetic fraud dataset and saves `model/saved_model.pkl`. If Kaggle credentials are configured, it downloads the real Credit Card Fraud dataset instead.

## Requirements

- Python 3.10+
- streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- imbalanced-learn >= 0.11.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- mlxtend >= 0.23.0

## License

MIT

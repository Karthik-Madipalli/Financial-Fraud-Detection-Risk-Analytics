# Financial Fraud Detection & Transaction Risk Analytics

A machine-learning project for detecting fraudulent financial transactions under severe class imbalance and converting model predictions into transaction-level risk scores.

## Overview

Financial fraud detection is a challenging classification problem because fraudulent transactions represent only a very small fraction of total transactions. In such settings, accuracy alone can provide a misleading assessment of model performance.

This project develops an end-to-end fraud detection and transaction risk analytics pipeline covering:

- Data quality analysis and duplicate removal
- Exploratory data analysis
- Feature transformation and preprocessing
- Supervised fraud classification
- Anomaly detection
- Model evaluation using imbalance-aware metrics
- Classification threshold optimization
- Transaction-level risk scoring
- Feature importance analysis
- False-positive and false-negative analysis

The final supervised model is a **Random Forest Classifier**, with its decision threshold selected using a validation set rather than relying on the default 0.50 threshold.
## Dataset

The project uses a financial transaction dataset containing anonymized transaction features and a binary fraud label.

The target variable is:

- `Class = 0` — Legitimate transaction
- `Class = 1` — Fraudulent transaction

After duplicate removal:

- **Transactions:** 283,726
- **Legitimate:** 283,253
- **Fraudulent:** 473
- **Fraud rate:** 0.1667%

The extreme class imbalance makes fraud recall, precision, F1-score, ROC-AUC, and particularly PR-AUC more informative than accuracy alone.

The anonymized transaction variables `V1–V28` are retained as modeling features, while `Time` and `Amount` are transformed into `Time_hours` and `LogAmount`.

## Project Workflow

The project follows an end-to-end fraud detection and transaction risk analytics workflow:

```text
Dataset
   ↓
Data Understanding
   ↓
Data Quality Analysis
   ↓
Exploratory Data Analysis
   ↓
Feature Analysis
   ↓
Data Preprocessing
   ↓
Model Development
   ↓
Model Evaluation
   ↓
Validation-Based Threshold Optimization
   ↓
Final Model Evaluation
   ↓
Transaction Risk Scoring
   ↓
Feature Importance Analysis
   ↓
Prediction Error Analysis
   ↓
Insights & Conclusions

## Machine Learning Models

The project evaluates both supervised classification and unsupervised anomaly detection approaches.

### 1. Baseline Logistic Regression

A standard Logistic Regression model is used as the initial supervised baseline.

It provides a linear decision boundary and establishes a reference point for evaluating more complex models.

### 2. Class-Weighted Logistic Regression

A second Logistic Regression model is trained using class weighting to explicitly address the severe class imbalance.

This approach increases the relative importance of fraudulent transactions during model training.

### 3. Random Forest Classifier

Random Forest is evaluated as the primary nonlinear supervised learning approach.

The model combines multiple decision trees and is capable of capturing nonlinear relationships and feature interactions that may not be represented by Logistic Regression.

The Random Forest decision threshold is subsequently optimized using a validation set to identify a suitable operating point for fraud detection.

### 4. Isolation Forest

Isolation Forest is evaluated as an anomaly-detection approach.

Unlike the supervised models, it does not directly learn the fraud labels during training. Instead, it identifies observations that appear anomalous relative to the overall transaction population.

This provides a comparison between supervised fraud classification and unsupervised anomaly detection.

## Model Performance

The models are evaluated using ROC-AUC, PR-AUC, fraud-class precision, fraud-class recall, and fraud-class F1-score.

| Model | ROC-AUC | PR-AUC | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 0.9532 | 0.7010 | 0.8485 | 0.5895 | 0.6957 |
| Class-Weighted Logistic Regression | 0.9625 | 0.6770 | 0.0554 | 0.8737 | 0.1042 |
| **Random Forest** | **0.9391** | **0.8011** | **0.9241** | **0.7684** | **0.8391** |
| Isolation Forest | 0.9329 | 0.1266 | 0.0300 | 0.7900 | 0.0600 |

Random Forest achieved the highest PR-AUC and the highest fraud-class F1-score among the evaluated approaches. Although the Logistic Regression models achieved higher ROC-AUC values, Random Forest provided a substantially stronger precision-recall balance for the minority fraud class.

The Class-Weighted Logistic Regression model achieved high fraud recall but produced very low fraud precision, resulting in a substantially lower F1-score. This demonstrates why model selection cannot rely on recall or ROC-AUC alone in a highly imbalanced fraud detection problem.

Isolation Forest achieved reasonable fraud recall but generated a large number of false positives, resulting in very low fraud precision and PR-AUC.

### Final Random Forest Performance

The Random Forest decision threshold was optimized using a validation set rather than selected arbitrarily at 0.50.

The selected operating threshold was **0.40**.

On the untouched test set, the final Random Forest achieved:

- **Fraud Precision:** 0.9333
- **Fraud Recall:** 0.7368
- **Fraud F1-score:** 0.8235
- **ROC-AUC:** 0.9391
- **PR-AUC:** 0.8011

The resulting confusion matrix was:

| | Predicted Legitimate | Predicted Fraudulent |
|---|---:|---:|
| **Actual Legitimate** | 56,646 | 5 |
| **Actual Fraudulent** | 25 | 70 |

Thus, the final model correctly identified **70 of 95 fraudulent transactions**, while generating only **5 false positives** among 56,651 legitimate transactions.

## Threshold Optimization

A classification threshold determines how predicted fraud scores are converted into binary fraud decisions.

Instead of assuming the conventional threshold of 0.50 is optimal, multiple thresholds were evaluated using the validation set.

The threshold was selected based on the **fraud-class F1-score**, which balances precision and recall.

The best validation result was obtained at:

- **Decision Threshold:** 0.40
- **Precision:** 0.9559
- **Recall:** 0.8553
- **F1-score:** 0.9028

The selected threshold of **0.40** was then fixed before evaluating the final model on the untouched test set.

This approach separates threshold optimization from final performance estimation and avoids selecting the operating threshold directly from the test data.

## Transaction Risk Scoring

Beyond binary fraud classification, the Random Forest model produces a fraud score for each transaction.

These scores are grouped into risk bands to support transaction-level prioritization:

| Probability Band | Risk Interpretation |
|---|---|
| 0–1% | Low Risk |
| 1–5% | Moderate Risk |
| 5–20% | Elevated Risk |
| 20–50% | High Risk |
| 50–75% | Very High Risk |
| 75–100% | Critical Risk |

The risk bands provide an operational interpretation of model output. Transactions with higher fraud scores can be prioritized for additional investigation rather than treating every transaction as equally suspicious.

The risk score should be interpreted as a **model-derived relative risk score**, not as a calibrated guarantee that a transaction is fraudulent.

## Error Analysis

The final Random Forest model produced 30 misclassified transactions on the test set:

- **False Positives (FP):** 5
- **False Negatives (FN):** 25

False positives are legitimate transactions incorrectly flagged as fraudulent, while false negatives are fraudulent transactions incorrectly classified as legitimate.

The error analysis shows an important asymmetry:

- False positives generally received relatively high fraud scores, indicating that some legitimate transactions exhibited patterns strongly associated with fraud by the model.
- False negatives generally received low fraud scores, indicating that several fraudulent transactions had feature patterns that were difficult for the model to distinguish from legitimate transactions.

Feature-level and transaction-amount analysis was performed on these misclassified cases to investigate the characteristics of the model's errors.

This analysis is important because, in practical fraud detection, understanding **why transactions are misclassified** is as important as reporting aggregate performance metrics.

## Key Insights

The analysis produced several important findings:

1. **Severe class imbalance is the central challenge.** Fraudulent transactions represent only a small fraction of the dataset, making accuracy an insufficient standalone evaluation metric.

2. **Class weighting improves fraud recall but can substantially increase false positives.** The class-weighted Logistic Regression model detected more fraudulent transactions but produced very low fraud precision.

3. **Random Forest provided the strongest precision-recall balance.** It achieved the highest PR-AUC and fraud-class F1-score among the evaluated approaches.

4. **Threshold selection materially affects operational performance.** Optimizing the Random Forest threshold on validation data provided a more appropriate operating point than relying on the default threshold of 0.50.

5. **Fraud risk is highly concentrated in the highest score ranges.** Transactions assigned very high fraud scores contain a disproportionately large share of the observed fraudulent transactions.

6. **False negatives remain an important limitation.** Some fraudulent transactions receive low model scores, demonstrating that certain fraud patterns overlap substantially with legitimate transaction behavior.

7. **Feature importance provides interpretability.** Features such as `V14`, `V10`, `V12`, `V17`, and `V4` were among the most influential features in the Random Forest model.

## Limitations

The current project has several limitations:

- The dataset contains anonymized features, which limits direct business interpretation of individual `V1–V28` variables.
- The dataset represents a historical transaction population and does not capture evolving fraud patterns or concept drift.
- Random Forest scores are model-derived risk scores and have not been treated as perfectly calibrated fraud probabilities.
- The current analysis is performed offline rather than in a production real-time transaction processing environment.
- False negatives remain possible, particularly for fraudulent transactions whose feature patterns resemble legitimate transactions.
- Threshold selection is based on F1-score; a production system could instead optimize an explicit business cost function that assigns different costs to false positives and false negatives.

## Future Improvements

Potential extensions of the project include:

- Cost-sensitive learning using explicit financial costs for false positives and false negatives
- Evaluation of gradient-boosting approaches such as XGBoost, LightGBM, or CatBoost
- SHAP-based model explainability for transaction-level predictions
- Temporal and behavioral features for detecting changes in transaction patterns
- Probability calibration for more reliable risk estimates
- Concept-drift detection and periodic model retraining
- Real-time inference through an API-based fraud detection service
- Human-in-the-loop investigation workflows for high-risk transactions
- Dynamic threshold recalibration based on changing fraud prevalence and operational costs

## Reproducibility

The complete analysis is contained in the Jupyter notebook:

`notebooks/financial_fraud_analysis.ipynb`

To reproduce the analysis:

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies.
4. Place the dataset in the expected `data/` directory.
5. Open the notebook in Jupyter Notebook or JupyterLab.
6. Run the notebook from top to bottom.

The notebook is designed to execute sequentially from a clean kernel and contains the complete data-processing, modelling, evaluation, and analysis workflow.

## Project Structure

```text
Financial-Fraud-Detection-Risk-Analytics/
│
├── data/
│   └── financial transaction dataset
│
├── notebooks/
│   └── financial_fraud_analysis.ipynb
│
├── src/
│   └── supporting source files
│
├── README.md
├── requirements.txt
└── .gitignore


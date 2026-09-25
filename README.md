# Supervised-Machine-Learning-for-Hospital-Readmission-Prediction

# Hospital Readmission Prediction

An end-to-end Machine Learning pipeline for predicting 30-day patient hospital readmissions. This project focuses on leak-free feature engineering, probability calibration, and cost-sensitive threshold optimization to minimize clinical False Negatives.

## Overview

Unplanned 30-day hospital readmissions are a critical quality metric in healthcare systems. Predicting readmission risks helps clinicians target high-risk patients with pre-discharge planning and follow-up care.

This project implements a robust machine learning pipeline designed to:
- Handle heterogeneous health data (numerical and categorical features).
- Prevent data leakage using scikit-learn Pipelines and ColumnTransformers.
- Evaluate multiple baseline algorithms under cross-validation.
- Calibrate output probabilities for downstream clinical risk scoring.
- Optimize classification thresholds using cost-sensitive scoring to prioritize high Recall/Sensitivity.

---

## Dataset Description

The dataset contains patient admission details, clinical indicators, and discharge information:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Integer | Patient age in years |
| `gender` | Categorical | Patient gender |
| `primary_diagnosis` | Categorical | Primary admission diagnosis (e.g., Heart Disease, Diabetes, COPD) |
| `num_procedures` | Integer | Number of medical procedures performed during stay |
| `days_in_hospital` | Integer | Total length of stay in days |
| `comorbidity_score` | Integer | Quantified score of co-occurring medical conditions |
| `discharge_to` | Categorical | Post-discharge location (e.g., Home, Home Health Care, SNF, Rehab) |
| `readmitted` | Binary (Target) | 30-day readmission status (1 = Readmitted, 0 = Not Readmitted) |

---

## Key Methodology

1. **Exploratory Data Analysis (EDA)**: Automated profiling (`ydata-profiling`) alongside univariate and bivariate analysis to inspect feature distributions and target imbalance (~19% positive rate).
2. **Leak-Free Preprocessing Pipeline**:
   - **Numerical**: Imputation (`SimpleImputer`) + Feature Scaling (`StandardScaler`, `RobustScaler`, `MinMaxScaler`).
   - **Categorical**: Categorical Imputation + One-Hot Encoding (`OneHotEncoder`).
3. **Model Selection**: Cross-validation benchmarking across diverse model families:
   - Logistic Regression
   - Decision Trees & Random Forests
   - Gradient Boosting / HistGradientBoosting / XGBoost
   - K-Nearest Neighbors, Naive Bayes, SVM
4. **Probability Calibration**: Diagnostic evaluation and calibration curves to ensure predicted probabilities match empirical readmission rates.
5. **Threshold Tuning**: Custom threshold optimization using `TunedThresholdClassifierCV` to balance Precision and Recall, specifically targeting cost reduction associated with undetected high-risk patients (False Negatives).

---

## Tech Stack

- **Language**: Python
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Profiling**: `ydata-profiling`
- **Machine Learning**: `scikit-learn`, `xgboost`

---





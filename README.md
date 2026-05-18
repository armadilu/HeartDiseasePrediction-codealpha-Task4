# Heart Disease Prediction

**CodeAlpha Machine Learning Internship — Task 4**

A machine learning project that predicts whether a patient is at risk of heart disease using clinical and demographic features. Four classification models are compared, and the most reliable one is recommended for screening use.

---

## Project Overview

The goal is to build a binary classifier that flags patients likely to have heart disease based on standard health indicators (age, chest pain type, cholesterol, blood pressure, ECG features, etc.). The model is intended as a screening aid that helps prioritize patients who need deeper clinical investigation.

---

## Dataset

Source: [UCI Heart Disease — Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

Combined dataset from 4 hospitals: **Cleveland, Hungary, Switzerland, and VA Long Beach.**

- 920 patients × 16 columns (raw)
- 920 patients × 11 columns (after cleaning)
- Target: `target` (1 = heart disease, 0 = no disease)
- Class balance: ~55% diseased / 45% healthy (no SMOTE required)

---

## Pipeline

1. Loaded and inspected the dataset
2. Dropped non-medical columns (`id`, `dataset`)
3. Converted `num` (0–4 severity) into a binary `target`
4. Handled missing values:
   - Dropped `ca`, `thal`, `slope` (>30% missing)
   - Filled numeric gaps with median
   - Filled categorical gaps with mode
5. Encoded text columns (`sex`, `cp`, `fbs`, `restecg`, `exang`) with LabelEncoder
6. Performed EDA — target distribution and correlation heatmap
7. Stratified 80/20 train-test split
8. Standardized features with StandardScaler
9. Trained 4 classifiers
10. Evaluated and compared with ROC curves
11. Extracted feature importance from the best model

---

## Models Used

- **Logistic Regression** — linear baseline
- **Decision Tree** — single-tree non-linear classifier
- **Random Forest** — 100-tree ensemble
- **XGBoost** — gradient boosting (industry standard)

---

## Results

| Model | Accuracy | ROC-AUC | Recall (Disease) |
|---|---|---|---|
| Logistic Regression | 0.82 | 0.899 | 0.86 |
| Decision Tree | 0.77 | 0.763 | 0.79 |
| Random Forest | 0.80 | 0.913 | 0.86 |
| **XGBoost** | **0.84** | 0.891 | **0.87** |

---

## Best Model: XGBoost

XGBoost was selected as the final model based on its highest accuracy (84%) and highest recall on the disease class (87%). In medical screening, recall is the most critical metric — missing a sick patient (false negative) carries a far higher cost than wrongly flagging a healthy one (false positive). Out of 102 actually sick patients in the test set, XGBoost correctly identified ~89.

Random Forest had the highest ROC-AUC (0.913) and is a strong alternative if calibrated probability outputs are preferred.

---

## Feature Importance

Top predictors of heart disease according to XGBoost:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Chest pain type (cp) | 32.6% |
| 2 | Sex | 15.0% |
| 3 | Exercise-induced angina (exang) | 11.7% |
| 4 | ST depression (oldpeak) | 6.8% |
| 5 | Cholesterol (chol) | 6.7% |

These rankings align with established clinical knowledge — chest pain type, exercise angina, and ST depression on ECG are all standard cardiovascular risk indicators.

---

## How to Run

1. Open `CodeAlpha_HeartDiseasePrediction.ipynb` in Google Colab.
2. Upload `heart_disease_uci.csv` to your Google Drive (or to the Colab session).
3. Run all cells via `Runtime → Run all`.

**Required libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost.

---

## Tech Stack

- Python 3.12
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- XGBoost
- Google Colab

---

## Author

**Aamina Khan**
CodeAlpha Machine Learning Internship — Task 4

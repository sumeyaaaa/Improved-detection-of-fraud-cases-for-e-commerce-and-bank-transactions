
# Improved Detection of Fraud Cases for E-Commerce and Bank Transactions

This project aims to develop machine learning models to **detect fraudulent transactions** in both e-commerce purchases and banking (credit card) contexts. Financial fraud is a growing concern for businesses and institutions; fraudulent transactions cost companies *millions* every year. In fact, global e-commerce fraud losses were projected to exceed $48 billion in 2023. By leveraging advanced ML techniques, we seek to identify fraud patterns in transactional data, reducing losses and improving security. This work is part of the 10 Academy AI Mastery program (Week 8 & 9, 2025), focusing on practical ML solutions for fraud prevention.

## Datasets

The analysis uses three primary datasets:

- **`Fraud_Data.csv`**: An e-commerce transactions dataset with user and transaction metadata.  
- **`IpAddress_to_Country.csv`**: A mapping file that assigns each numerical IP address range to a country.  
- **`creditcard.csv`**: A standard credit card transaction dataset with anonymized features.  

## ✅ Current Progress

- **Data Loading & Preprocessing:** Data loaded, cleaned, and merged (including IP to country).
- **Feature Engineering:** Time-based features like `time_since_signup`, `hour_of_day`, `day_of_week` extracted.
- **Class Imbalance Handling:** Applied SMOTE on training data only.
- **EDA:** Explored fraud distribution, source-wise behavior, and age group tendencies.
- **Model Training:** Trained Logistic Regression, XGBoost, LightGBM, and CatBoost on both datasets.
- **Evaluation:** Compared models using F1 Score and PR AUC.
- **Model Selection:** XGBoost selected as best for both datasets.
- **Model Saving:** Saved final models as `.pkl` for deployment.

## 📊 Model Comparison Results

### 🛒 E-commerce Fraud Dataset (`Fraud_Data.csv`)

| Model               | F1 Score | PR AUC | Comments |
|--------------------|----------|--------|----------|
| Logistic Regression | 0.2723   | 0.3877 | Poor recall and high false positives |
| **XGBoost**          | **0.6903**   | **0.6240** | Best precision-recall balance |
| LightGBM           | 0.6901   | 0.6216 | Nearly tied with XGBoost |
| CatBoost           | 0.6900   | 0.6218 | Slightly behind |

### 💳 Credit Card Fraud Dataset (`creditcard.csv`)

| Model               | F1 Score | PR AUC | Comments |
|--------------------|----------|--------|----------|
| Logistic Regression | 0.0993   | 0.6763 | Weak at detecting frauds |
| **XGBoost**          | **0.7638**   | **0.8085** | Best F1 score and excellent PR AUC |
| LightGBM           | 0.6838   | **0.8177** | Best PR AUC, but lower F1 score |
| CatBoost           | 0.6912   | 0.7933 | Strong performance, but not best overall |

## 🧠 Explainability (SHAP)

- Initialized SHAP on `XGBoost` models for both datasets.
- Generated:
  - **Summary plots** to visualize global feature importance.
  - **Force plots** to explain individual fraud predictions.
- Key insights: time-based features and anonymized PCA components (in credit data) drive fraud classification.

## 🔄 Planned Work

- **Complete SHAP Explainability:** Finish and interpret plots for both models.
- **Final Reporting:** Prepare README, blog post, and interim/final reports.
- **Optional Deployment:** Expose model using Flask or Dash for inference API.



## Tools and Libraries

Python 3.x, pandas, numpy, scikit-learn, XGBoost, imbalanced-learn, SHAP, matplotlib, seaborn, Flask/Dash.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

- Run notebooks in `notebooks/` for EDA and model training.
- Use `scripts/` for preprocessing, model training, SHAP analysis.
- Flask/Dash app (optional) via `app.py`.

## Project Structure

```
.
├── data/
├── notebooks/
├── scripts/
├── outputs/
├── requirements.txt
└── README.md
```

## Credits and References

Kaggle datasets, SHAP documentation, DataCamp tutorials, and 10 Academy resources.


# Improved Detection of Fraud Cases for E-Commerce and Bank Transactions

This project aims to develop machine learning models to **detect fraudulent transactions** in both e-commerce purchases and banking (credit card) contexts. Financial fraud is a growing concern for businesses and institutions; fraudulent transactions cost companies *millions* every year. In fact, global e-commerce fraud losses were projected to exceed $48 billion in 2023. By leveraging advanced ML techniques, we seek to identify fraud patterns in transactional data, reducing losses and improving security. This work is part of the 10 Academy AI Mastery program (Week 8 & 9, 2025), focusing on practical ML solutions for fraud prevention.

## Datasets

The analysis uses three primary datasets:

- **`Fraud_Data.csv`**: An e-commerce transactions dataset with user and transaction metadata.  
- **`IpAddress_to_Country.csv`**: A mapping file that assigns each numerical IP address range to a country.  
- **`creditcard.csv`**: A standard credit card transaction dataset with anonymized features.  

## Current Progress

- **Data Loading & Preprocessing:** Data loading, merging IP-country data, and encoding.
- **Feature Engineering:** Features like `time_since_signup`, `transaction_hour`, etc.
- **Class Imbalance Handling:** Techniques like SMOTE applied.
- **EDA:** Key insights on fraud patterns.
- **Model Building Plan:** Logistic Regression and XGBoost models prepared for training.

## Planned Work

- **Model Training & Evaluation:** Use of metrics like AUC-PR and F1.
- **Explainability (SHAP):** SHAP plots for global and local interpretability.
- **Model Selection & Interpretation**
- **Deployment (Optional):** Using Flask or Dash.

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

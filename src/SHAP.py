# shap_explainability.py

"""
This script demonstrates how to reload processed test data, apply the saved preprocessor,
load the best trained model, and generate SHAP explanations for global and local interpretability.
It focuses on the e-commerce dataset (Fraud_Data.csv) and supports task-3 interpretability goals.
"""

import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import joblib

# 1. Reload and preprocess test data (e-commerce fraud dataset)
fraud_df["signup_time"] = pd.to_datetime(fraud_df["signup_time"])
fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])
fraud_df["time_since_signup"] = (fraud_df["purchase_time"] - fraud_df["signup_time"]).dt.total_seconds() / 3600
fraud_df["hour_of_day"] = fraud_df["purchase_time"].dt.hour
fraud_df["day_of_week"] = fraud_df["purchase_time"].dt.dayofweek

# Drop unused columns for modeling
fraud_df.drop(columns=["user_id", "device_id", "ip_address", "signup_time", "purchase_time"], inplace=True)

# Separate features and target
X = fraud_df.drop(columns=["class"])
y = fraud_df["class"]

# Match original split for SHAP explanations
X_train_ecom, X_test_ecom, y_train_ecom, y_test_ecom = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Load preprocessor and model
preprocessor = joblib.load("preprocessor_ecom.pkl")
best_model = joblib.load("best_model_ecommerce.pkl")

# 3. Transform test data using saved pipeline
X_test_transformed = preprocessor.transform(X_test_ecom)

# 4. Generate SHAP values using TreeExplainer
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_transformed)

# 5. Optional Visualization
# shap.summary_plot(shap_values, features=X_test_transformed, feature_names=preprocessor.get_feature_names_out())
# shap.force_plot(explainer.expected_value, shap_values[0], X_test_transformed[0], matplotlib=True)

# The SHAP output can be used for:
# - global insights (feature importance)
# - local inspection of specific fraud decisions

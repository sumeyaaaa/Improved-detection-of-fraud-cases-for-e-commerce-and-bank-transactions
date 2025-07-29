# train_evaluate_save_models.py

"""
This module handles:
- Training and evaluation of four classifiers: Logistic Regression, XGBoost, LightGBM, CatBoost
- Evaluation using F1 Score and PR AUC
- Model selection logic using F1 as primary and PR AUC as tie-breaker
- Saving all models including the best per dataset for reuse/deployment

Used for both e-commerce (Fraud_Data.csv) and credit card fraud (creditcard.csv) datasets.
"""

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42

# Evaluation function

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """Trains the model and evaluates it on test set, returning F1, PR AUC, and Confusion Matrix."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name} -- F1 Score: {f1:.4f},  PR AUC: {pr_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return f1, pr_auc, cm

# ---- Model Training and Selection for E-commerce ---- #

print("**E-commerce Fraud Dataset (Fraud_Data.csv)**")

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
lgb_clf = LGBMClassifier(random_state=RANDOM_STATE)
cat_clf = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)

f1_lr_e,  pr_lr_e,  cm_lr_e  = train_and_evaluate(log_reg,  X_train_ecom_res, y_train_ecom_res, X_test_ecom_proc, y_test_ecom, "Logistic Regression")
f1_xgb_e, pr_xgb_e, cm_xgb_e = train_and_evaluate(xgb_clf, X_train_ecom_res, y_train_ecom_res, X_test_ecom_proc, y_test_ecom, "XGBoost Classifier")
f1_lgb_e, pr_lgb_e, cm_lgb_e = train_and_evaluate(lgb_clf, X_train_ecom_res, y_train_ecom_res, X_test_ecom_proc, y_test_ecom, "LightGBM Classifier")
f1_cat_e, pr_cat_e, cm_cat_e = train_and_evaluate(cat_clf, X_train_ecom_res, y_train_ecom_res, X_test_ecom_proc, y_test_ecom, "CatBoost Classifier")

best_model_name_e = None
best_f1_e = -1; best_pr_e = -1; best_model_obj_e = None
for model_name, f1, pr, model_obj in [
    ("Logistic Regression",   f1_lr_e,  pr_lr_e,  log_reg),
    ("XGBoost Classifier",    f1_xgb_e, pr_xgb_e, xgb_clf),
    ("LightGBM Classifier",   f1_lgb_e, pr_lgb_e, lgb_clf),
    ("CatBoost Classifier",   f1_cat_e, pr_cat_e, cat_clf)]:
    if f1 > best_f1_e or (f1 == best_f1_e and pr > best_pr_e):
        best_model_name_e = model_name
        best_f1_e = f1
        best_pr_e = pr
        best_model_obj_e = model_obj

print(f"\nBest model for E-commerce fraud data: {best_model_name_e} (F1={best_f1_e:.4f}, PR AUC={best_pr_e:.4f})")

# ---- Model Training and Selection for Credit Card ---- #

print("\n**Credit Card Fraud Dataset (creditcard.csv)**")

log_reg_cc = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
xgb_cc     = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
lgb_cc     = LGBMClassifier(random_state=RANDOM_STATE)
cat_cc     = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)

f1_lr_c,  pr_lr_c,  cm_lr_c  = train_and_evaluate(log_reg_cc, X_train_cc_res, y_train_cc_res, X_test_cc_proc, y_test_cc, "Logistic Regression")
f1_xgb_c, pr_xgb_c, cm_xgb_c = train_and_evaluate(xgb_cc,    X_train_cc_res, y_train_cc_res, X_test_cc_proc, y_test_cc, "XGBoost Classifier")
f1_lgb_c, pr_lgb_c, cm_lgb_c = train_and_evaluate(lgb_cc,    X_train_cc_res, y_train_cc_res, X_test_cc_proc, y_test_cc, "LightGBM Classifier")
f1_cat_c, pr_cat_c, cm_cat_c = train_and_evaluate(cat_cc,    X_train_cc_res, y_train_cc_res, X_test_cc_proc, y_test_cc, "CatBoost Classifier")

best_model_name_c = None
best_f1_c = -1; best_pr_c = -1; best_model_obj_c = None
for model_name, f1, pr, model_obj in [
    ("Logistic Regression",   f1_lr_c,  pr_lr_c,  log_reg_cc),
    ("XGBoost Classifier",    f1_xgb_c, pr_xgb_c, xgb_cc),
    ("LightGBM Classifier",   f1_lgb_c, pr_lgb_c, lgb_cc),
    ("CatBoost Classifier",   f1_cat_c, pr_cat_c, cat_cc)]:
    if f1 > best_f1_c or (f1 == best_f1_c and pr > best_pr_c):
        best_model_name_c = model_name
        best_f1_c = f1
        best_pr_c = pr
        best_model_obj_c = model_obj

print(f"\nBest model for Credit Card fraud data: {best_model_name_c} (F1={best_f1_c:.4f}, PR AUC={best_pr_c:.4f})")

# Save all models
joblib.dump(log_reg,    "logistic_ecom_model.pkl")
joblib.dump(xgb_clf,    "xgb_ecom_model.pkl")
joblib.dump(lgb_clf,    "lgbm_ecom_model.pkl")
joblib.dump(cat_clf,    "catboost_ecom_model.pkl")

joblib.dump(log_reg_cc, "logistic_credit_model.pkl")
joblib.dump(xgb_cc,     "xgb_credit_model.pkl")
joblib.dump(lgb_cc,     "lgbm_credit_model.pkl")
joblib.dump(cat_cc,     "catboost_credit_model.pkl")

# Save best model from each dataset
joblib.dump(best_model_obj_e, "best_model_ecommerce.pkl")
joblib.dump(best_model_obj_c, "best_model_creditcard.pkl")

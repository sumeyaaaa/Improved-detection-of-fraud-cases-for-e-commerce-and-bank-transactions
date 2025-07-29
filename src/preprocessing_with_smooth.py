# preprocessing_with_smote.py

"""
This script defines a reusable function `preprocess_with_smote` to prepare the training and test data
by applying preprocessing and SMOTE-based resampling. It ensures:

- Missing values are handled appropriately.
- Categorical features are one-hot encoded.
- Numeric features are scaled.
- Class imbalance in the training set is corrected using SMOTE.

The function returns transformed and resampled training data,
processed test data, and the fitted preprocessing pipeline for deployment or SHAP.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

def preprocess_with_smote(X_train, X_test, y_train):
    """
    Fits a preprocessing pipeline on X_train and transforms X_train and X_test.
    The pipeline includes missing value imputation, one-hot encoding for categoricals,
    and scaling for numerics. After transforming, applies SMOTE to X_train to balance classes.

    Parameters:
    - X_train: Training features DataFrame
    - X_test: Test features DataFrame
    - y_train: Training target Series

    Returns:
    - X_train_res: Resampled training features
    - y_train_res: Resampled training labels
    - X_test_proc: Transformed test features
    - preprocessor: Fitted ColumnTransformer for reuse
    """
    # Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool', 'int32']).columns.tolist()

    # Reclassify time-derived int features as categorical
    if 'day_of_week' in numeric_features:
        numeric_features.remove('day_of_week')
        categorical_features.append('day_of_week')
    if 'hour_of_day' in numeric_features:
        numeric_features.remove('hour_of_day')
        categorical_features.append('hour_of_day')

    # Numeric pipeline: impute with median, scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute with constant, then one-hot encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both into a single preprocessor
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    # Fit preprocessor on training data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(f"Preprocessing: {X_train.shape[1]} features -> {X_train_proc.shape[1]} features after encoding/scaling.")

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    print("Applied SMOTE: Training samples before =", X_train_proc.shape[0], ", after =", X_train_res.shape[0])
    print(f"Fraud class percentage after SMOTE: {100 * y_train_res.mean():.1f}% (should be ~50%)")

    return X_train_res, y_train_res, X_test_proc, preprocessor

# Example Usage:
# X_train_ecom_res, y_train_ecom_res, X_test_ecom_proc, preproc_ecom = preprocess_with_smote(X_train_ecom, X_test_ecom, y_train_ecom)
# X_train_cc_res, y_train_cc_res, X_test_cc_proc, preproc_cc = preprocess_with_smote(X_train_cc, X_test_cc, y_train_cc)

# Save preprocessor objects
# import joblib
# joblib.dump(preproc_ecom, "preprocessor_ecom.pkl")
# joblib.dump(preproc_cc, "preprocessor_cc.pkl")

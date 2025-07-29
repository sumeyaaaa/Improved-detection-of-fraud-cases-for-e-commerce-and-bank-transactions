# splitting_data.py

"""
This script defines a reusable function `prepare_and_split_data` to clean, engineer features,
and split the data for both e-commerce and credit card fraud detection datasets.

The function handles:
- Duplicate removal
- Date parsing and time-based feature engineering (e-commerce only)
- Dropping unused or high-cardinality identifiers
- Train/test split with stratification

Used for both: 
- Fraud_Data.csv (e-commerce dataset)
- creditcard.csv (bank dataset)
"""

from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_STATE = 42

def prepare_and_split_data(df, target_col, is_ecommerce=False):
    """
    Cleans and preprocesses the raw dataframe:
      - Removes duplicates
      - Parses dates and creates time-based features (for e-commerce data)
      - Drops unused identifier columns
      - Splits into train/test sets.

    Parameters:
    - df: pandas DataFrame
    - target_col: name of target column ('class' or 'Class')
    - is_ecommerce: True if processing e-commerce dataset (Fraud_Data.csv)

    Returns:
    - X_train, X_test, y_train, y_test
    """
    df = df.copy()  # avoid modifying original data

    # 1. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 2. Feature engineering for e-commerce data
    if is_ecommerce:
        # Parse datetime fields
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Time delta between signup and purchase (in hours)
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0

        # Extract temporal features
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek  # 0=Monday

        # Drop high-cardinality/unnecessary columns
        df.drop(['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address'], axis=1, inplace=True)
    else:
        # Drop 'Time' for creditcard.csv (anonymized time delta, not useful as-is)
        df.drop(['Time'], axis=1, inplace=True)

    # 3. Split into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Split '{target_col}' -> Train size: {len(X_train)}, Test size: {len(X_test)}, Fraud% in train: {100*y_train.mean():.3f}%")

    return X_train, X_test, y_train, y_test

# Example Usage:
# X_train_ecom, X_test_ecom, y_train_ecom, y_test_ecom = prepare_and_split_data(fraud_df, target_col='class', is_ecommerce=True)
# X_train_cc, X_test_cc, y_train_cc, y_test_cc = prepare_and_split_data(credit_df, target_col='Class', is_ecommerce=False)

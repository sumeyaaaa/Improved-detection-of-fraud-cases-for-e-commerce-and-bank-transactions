from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

def handle_class_imbalance(X, y, random_state=42):
    """
    Apply SMOTE to balance classes in the training set.

    Returns:
    X_resampled, y_resampled
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    print("Before SMOTE:", y.value_counts().to_dict())
    print("After SMOTE:", pd.Series(y_res).value_counts().to_dict())
    return X_res, y_res


def scale_numerical_features(df, columns):
    """
    Apply StandardScaler to specified numerical columns.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def encode_categorical_features(df, columns):
    """
    Apply One-Hot Encoding to specified categorical columns.
    """
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

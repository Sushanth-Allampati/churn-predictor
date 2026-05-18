# src/features.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ── Column definitions ────────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'charges_per_month',   # derived — added in engineer_features()
    'num_services',        # derived — added in engineer_features()
]

BINARY_FEATURES = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'SeniorCitizen',       # already 0/1 in raw data, included for completeness
]

MULTI_CAT_FEATURES = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod',
]

TARGET = 'Churn'
DROP_COLS = ['customerID']

# Binary columns that need Yes/No → 1/0 encoding
# (SeniorCitizen is already int, so excluded here)
BINARY_YES_NO = [
    'gender',        # Female=0, Male=1
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
]

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV and perform dtype fixes that are properties
    of the data itself, not transformations that risk leakage.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with corrected dtypes.
    """
    df = pd.read_csv(path)

    # Fix TotalCharges: whitespace strings → NaN → 0.0
    # The 11 affected rows are new customers (tenure=0) with no bill yet.
    # Imputing 0 is correct here — it's not a missing value, it's a real zero.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps that are safe to do before the train/test split:
    - Drop identifier columns
    - Encode binary Yes/No columns to 1/0
    - Encode target column to 1/0

    Does NOT apply scaling or one-hot encoding — those go inside
    the sklearn Pipeline to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_raw_data().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for splitting.
    """
    df = df.copy()   # never mutate the input

    # Drop identifier — carries no signal
    df = df.drop(columns=DROP_COLS)

    # Encode binary Yes/No columns
    for col in BINARY_YES_NO:
        df[col] = (df[col].str.strip()
                           .map({'Yes': 1, 'No': 1,    # placeholder
                                 'Female': 0, 'Male': 1,
                                 'No': 0, 'Yes': 1})
                  )

    # Cleaner approach — map each column explicitly
    yes_no_map = {'Yes': 1, 'No': 0}
    gender_map = {'Male': 1, 'Female': 0}

    df['gender']          = df['gender'].map(gender_map)
    df['Partner']         = df['Partner'].map(yes_no_map)
    df['Dependents']      = df['Dependents'].map(yes_no_map)
    df['PhoneService']    = df['PhoneService'].map(yes_no_map)
    df['PaperlessBilling']= df['PaperlessBilling'].map(yes_no_map)

    # Encode target
    df[TARGET] = df[TARGET].map(yes_no_map)

    # Verify no NaNs introduced by the mapping
    introduced_nulls = df[BINARY_YES_NO + [TARGET]].isnull().sum()
    if introduced_nulls.any():
        raise ValueError(
            f"Encoding introduced nulls — check raw data:\n{introduced_nulls}"
        )

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from existing columns.
    Safe to apply before splitting — no statistics from the data are used,
    only deterministic arithmetic.

    New features
    ------------
    charges_per_month : float
        TotalCharges / (tenure + 1). Normalises total spend by tenure,
        removing the multicollinearity between TotalCharges and tenure.
        +1 prevents division by zero for new customers (tenure=0).

    num_services : int
        Count of add-on services the customer subscribes to.
        Captures the consistent churn-reduction pattern seen in EDA
        without needing six separate binary columns.
    """
    df = df.copy()

    # Derived feature 1: cost per month of tenure
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Derived feature 2: number of add-on services
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    # These columns have values: 'Yes', 'No', 'No internet service'
    # Count only 'Yes' values
    df['num_services'] = (
        df[service_cols]
        .apply(lambda col: (col == 'Yes').astype(int))
        .sum(axis=1)
    )

    return df

from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame,
               val_size:  float = 0.15,
               test_size: float = 0.15,
               random_state: int = 42):
    """
    Stratified train / val / test split.

    Stratification ensures the 26.5% churn rate is preserved
    in all three splits — important for imbalanced datasets.

    Parameters
    ----------
    df           : cleaned + engineered dataframe
    val_size     : fraction of total data for validation
    test_size    : fraction of total data for test
    random_state : seed for reproducibility

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # First split: hold out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: split remaining into train + val
    # val_size is relative to the full dataset, so adjust for remaining fraction
    adjusted_val = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val,
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
    - Applies StandardScaler to numerical features
    - Applies OneHotEncoder to multi-category features
    - Passes binary features through unchanged (already 0/1)

    This is fit ONLY on training data and applied to val/test,
    preventing any leakage of val/test statistics.

    Returns
    -------
    sklearn ColumnTransformer (unfitted)
    """
    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',   # unseen categories → all zeros (safe for prod)
        sparse_output=False,       # return dense array (easier to inspect)
        drop='first',              # drop first level to avoid dummy variable trap
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, MULTI_CAT_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES),
        ],
        remainder='drop'           # drop any columns not listed above
    )

    return preprocessor

import os

def run_pipeline(raw_path: str, processed_dir: str = None):
    """
    Full data preparation pipeline:
    load → clean → engineer → split → (optionally save splits)

    Parameters
    ----------
    raw_path      : path to raw CSV
    processed_dir : if provided, saves X/y splits as CSVs here

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
        X_train.to_csv(f'{processed_dir}/X_train.csv', index=False)
        X_val.to_csv(  f'{processed_dir}/X_val.csv',   index=False)
        X_test.to_csv( f'{processed_dir}/X_test.csv',  index=False)
        y_train.to_csv(f'{processed_dir}/y_train.csv', index=False)
        y_val.to_csv(  f'{processed_dir}/y_val.csv',   index=False)
        y_test.to_csv( f'{processed_dir}/y_test.csv',  index=False)
        print(f"Splits saved to {processed_dir}/")

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.3f}")
    print(f"Val churn rate:   {y_val.mean():.3f}")
    print(f"Test churn rate:  {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


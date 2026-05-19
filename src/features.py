# src/features.py

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ── Column definitions ────────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'charges_per_month',
    'num_services',
]

BINARY_FEATURES = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'SeniorCitizen',
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

BINARY_YES_NO = [
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
]


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_raw_data(path: str) -> pd.DataFrame:
    """Read raw CSV and fix TotalCharges dtype."""
    df = pd.read_csv(path)

    # TotalCharges has whitespace strings for 11 new customers (tenure=0)
    # Coerce to float, fill the 11 NaNs with 0 (correct business value)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    return df


# ── Step 2: Clean ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifiers, encode binary Yes/No columns to 1/0,
    encode target Churn to 1/0.

    Does NOT scale or one-hot encode — those go inside the
    sklearn Pipeline to prevent data leakage.
    """
    df = df.copy()

    # Drop customer ID — no signal
    df = df.drop(columns=DROP_COLS)

    # Encode gender
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Encode all Yes/No binary columns
    yes_no_map = {'Yes': 1, 'No': 0}
    for col in BINARY_YES_NO:
        df[col] = df[col].map(yes_no_map)

    # Encode target
    df[TARGET] = df[TARGET].map(yes_no_map)

    # Guard: crash loudly if any encoding produced NaNs
    cols_to_check = ['gender'] + BINARY_YES_NO + [TARGET]
    null_counts = df[cols_to_check].isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Encoding introduced nulls — check raw data:\n{null_counts}"
        )

    return df


# ── Step 3: Engineer features ─────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create two derived features:
    - charges_per_month : removes multicollinearity between TotalCharges and tenure
    - num_services      : captures add-on service count as a single column
    """
    df = df.copy()

    # Cost normalised by tenure — +1 prevents division by zero for new customers
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Count of add-on services the customer subscribes to (0–6)
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    df['num_services'] = (
        df[service_cols]
        .apply(lambda col: (col == 'Yes').astype(int))
        .sum(axis=1)
    )

    return df


# ── Step 4: Split ─────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               val_size: float = 0.15,
               test_size: float = 0.15,
               random_state: int = 42):
    """
    Stratified train / val / test split.
    Stratification preserves the 26.5% churn rate in all three splits.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Hold out test set first
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Split remainder into train + val
    adjusted_val = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Step 5: Build preprocessor ────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build an unfitted sklearn ColumnTransformer:
    - StandardScaler     → numerical features
    - OneHotEncoder      → multi-category features
    - passthrough        → binary features (already 0/1)

    Fit ONLY on training data. Transform val and test using
    the stored training statistics — never refit on val/test.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                'cat',
                OneHotEncoder(
                    handle_unknown='ignore',  # unseen categories → all zeros
                    sparse_output=False,      # dense array (sklearn >= 1.2)
                    drop='first',             # avoid dummy variable trap
                ),
                MULTI_CAT_FEATURES,
            ),
            (
                'bin',
                'passthrough',
                BINARY_FEATURES,
            ),
        ],
        remainder='drop',
    )

    return preprocessor


# ── Step 6: Full pipeline convenience function ────────────────────────────────

def run_pipeline(raw_path: str, processed_dir: str = None):
    """
    Run the full data preparation sequence:
    load → clean → engineer → split → (optionally save splits to disk)

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

    print(f"Train : {X_train.shape} | churn rate: {y_train.mean():.3f}")
    print(f"Val   : {X_val.shape}   | churn rate: {y_val.mean():.3f}")
    print(f"Test  : {X_test.shape}  | churn rate: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test
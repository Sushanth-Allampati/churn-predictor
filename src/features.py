# src/features.py
"""
src/features.py
───────────────
Data preparation pipeline for the Telco Customer Churn predictor.

Pipeline sequence (call in this order):
    1. load_raw_data()      — read CSV, fix TotalCharges dtype
    2. clean_data()         — drop ID, encode binary columns and target
    3. engineer_features()  — create charges_per_month and num_services
    4. split_data()         — stratified train / val / test split
    5. build_preprocessor() — build unfitted ColumnTransformer
    6. run_pipeline()       — convenience wrapper for steps 1–4

The ColumnTransformer from build_preprocessor() is fit inside
src/train.py on training data only — never here.

Column constants (NUMERICAL_FEATURES, BINARY_FEATURES, etc.) are
the single source of truth for column names across the entire project.
Any other file that needs column lists imports them from here.
"""
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
    """
    Read the raw Telco CSV and fix the TotalCharges column dtype.

    TotalCharges is stored as object in the raw file because 11 rows
    contain whitespace strings instead of numbers. These 11 customers
    have tenure=0 (brand new, not yet billed) so the correct imputation
    is 0.0, not the column mean.

    Parameters
    ----------
    path : str
        Path to the raw CSV file (e.g. 'data/raw/telco_churn.csv').

    Returns
    -------
    pd.DataFrame
        Raw dataframe with TotalCharges as float64, 7043 rows, 21 columns.
    """
    df = pd.read_csv(path)

    # TotalCharges has whitespace strings for 11 new customers (tenure=0)
    # Coerce to float, fill the 11 NaNs with 0 (correct business value)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    return df


# ── Step 2: Clean ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps that are safe before the train/test split.

    Specifically:
    - Drops customerID (random identifier, zero predictive signal)
    - Encodes gender to 1/0 (Male=1, Female=0)
    - Encodes Yes/No binary columns to 1/0
    - Encodes the target Churn to 1/0

    Does NOT apply StandardScaler or OneHotEncoder. Those transformations
    use statistics computed from the data (mean, std, category frequencies)
    and must be fit only on training data to prevent leakage. They belong
    inside the sklearn Pipeline in src/train.py.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_raw_data().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, 7043 rows, 20 columns (customerID dropped,
        Churn encoded to int).

    Raises
    ------
    ValueError
        If any binary encoding produces NaN — signals unexpected raw data.
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
    Create two derived features from existing columns.

    charges_per_month
        Formula: TotalCharges / (tenure + 1)
        Motivation: TotalCharges and tenure have 0.83 correlation in this
        dataset — they carry nearly redundant information. This feature
        decouples them by normalising total spend by tenure length.
        The +1 prevents division by zero for new customers (tenure=0).
        Confirmed stronger point-biserial correlation with Churn than
        raw TotalCharges in EDA (see notebooks/01_eda.ipynb).

    num_services
        Formula: count of 'Yes' values across 6 add-on service columns
        Motivation: all 6 service columns (OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies) show
        the same directional relationship with Churn in EDA. Collapsing
        them into one count feature replaces 12 one-hot encoded columns
        with 1 numerical column, reducing overfitting risk.

    Both features are derived arithmetically — no statistics from the
    data are used — so they are safe to compute before splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Output of clean_data().

    Returns
    -------
    pd.DataFrame
        Input dataframe plus two new columns: charges_per_month (float)
        and num_services (int, range 0–6).
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
    Stratified three-way split: train / val / test.

    Stratification preserves the 26.5% churn rate in all three splits.
    Without stratification, a random split could produce a val set with
    20% or 33% churn by chance, making val metrics an unreliable guide
    for model selection.

    The split is done in two stages:
        Stage 1: hold out test_size fraction as the test set
        Stage 2: split the remainder into train and val, adjusting
                 val_size to be relative to the full dataset size

    The test set is never used during training or hyperparameter tuning.
    It is touched exactly once — for the final reported metrics.

    Parameters
    ----------
    df           : pd.DataFrame — output of engineer_features()
    val_size     : float — fraction of total data for validation (default 0.15)
    test_size    : float — fraction of total data for test (default 0.15)
    random_state : int   — random seed for reproducibility (default 42)

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series
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
    Build and return an unfitted sklearn ColumnTransformer.

    Transformations applied:
    ┌─────────────────────┬──────────────────┬─────────────────────────┐
    │ Transformer         │ Columns          │ Reason                  │
    ├─────────────────────┼──────────────────┼─────────────────────────┤
    │ StandardScaler      │ NUMERICAL (5)    │ XGBoost is scale-       │
    │                     │                  │ invariant but LR needs  │
    │                     │                  │ scaled features         │
    │ OneHotEncoder       │ MULTI_CAT (10)   │ Nominal categories with │
    │ drop='first'        │                  │ no ordinal relationship  │
    │ handle_unknown=     │                  │ handle_unknown prevents  │
    │ 'ignore'            │                  │ API crash on new values │
    │ passthrough         │ BINARY (6)       │ Already 0/1 — scaling   │
    │                     │                  │ would change nothing    │
    └─────────────────────┴──────────────────┴─────────────────────────┘

    Returns an UNFITTED transformer. Call .fit_transform(X_train) in
    src/train.py — never fit on val or test data.

    Returns
    -------
    sklearn.compose.ColumnTransformer (unfitted)
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
    Convenience wrapper: runs the full data preparation sequence.

    Calls in order: load_raw_data → clean_data → engineer_features → split_data.
    Optionally saves all six splits (X/y for train/val/test) to disk as CSVs.

    Use this function in:
    - Notebooks for quick data loading
    - src/train.py to get splits before training
    - tests/ fixtures to load data once per test module

    Do NOT use this to load data inside the API. The API receives single
    rows of pre-split data at inference time — it should call
    build_preprocessor() with a pre-fitted transformer loaded from disk.

    Parameters
    ----------
    raw_path      : str  — path to raw CSV file
    processed_dir : str  — if provided, saves splits here as CSV files.
                           Creates the directory if it doesn't exist.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
    y_train, y_val, y_test : pd.Series
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
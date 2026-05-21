# tests/test_features.py

import pytest
import numpy as np
import pandas as pd
from src.features import (
    run_pipeline,
    build_preprocessor,
    NUMERICAL_FEATURES,
    MULTI_CAT_FEATURES,
    BINARY_FEATURES,
)

RAW_PATH = 'data/raw/telco_churn.csv'


@pytest.fixture(scope='module')
def splits():
    """Load and split data once for all tests in this module."""
    return run_pipeline(RAW_PATH)


@pytest.fixture(scope='module')
def transformed(splits):
    """Fit preprocessor on train, transform all splits."""
    X_train, X_val, X_test, *_ = splits
    pp = build_preprocessor()
    X_tr = pp.fit_transform(X_train)
    X_v  = pp.transform(X_val)
    X_te = pp.transform(X_test)
    return pp, X_tr, X_v, X_te


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_split_shapes(splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    assert X_train.shape[0] > X_val.shape[0]
    assert X_train.shape[0] > X_test.shape[0]
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


def test_transformed_shapes_consistent(transformed):
    _, X_tr, X_v, X_te = transformed
    assert X_tr.shape[1] == X_v.shape[1] == X_te.shape[1], \
        "All splits must have same number of columns after transform"


# ── Leakage tests ─────────────────────────────────────────────────────────────

def test_stratified_churn_rate(splits):
    """All splits must have approximately the same churn rate."""
    _, _, _, y_train, y_val, y_test = splits
    rates = [y_train.mean(), y_val.mean(), y_test.mean()]
    assert max(rates) - min(rates) < 0.02, \
        f"Churn rates diverge too much: {rates}"


def test_no_target_in_features(splits):
    """Churn column must not appear in X."""
    X_train, *_ = splits
    assert 'Churn' not in X_train.columns


def test_no_customer_id_in_features(splits):
    """customerID must be dropped."""
    X_train, *_ = splits
    assert 'customerID' not in X_train.columns


# ── Quality tests ─────────────────────────────────────────────────────────────

def test_no_nan_after_transform(transformed):
    """Transformed arrays must contain no NaN values."""
    _, X_tr, X_v, X_te = transformed
    for name, arr in [('train', X_tr), ('val', X_v), ('test', X_te)]:
        assert not np.isnan(arr).any(), \
            f"NaN found in {name} after transform"


def test_numerical_scaling_on_train(transformed):
    """Numerical features on train should have mean≈0, std≈1 after scaling."""
    _, X_tr, _, _ = transformed
    n = len(NUMERICAL_FEATURES)
    num_block = X_tr[:, :n]
    assert abs(num_block.mean()) < 0.01, \
        "Train numerical mean should be ≈ 0"
    assert abs(num_block.std() - 1.0) < 0.1, \
        "Train numerical std should be ≈ 1"


def test_binary_values_unchanged(transformed):
    """Binary passthrough columns should only contain 0 and 1."""
    pp, X_tr, _, _ = transformed
    cat_features = pp.named_transformers_['cat'].get_feature_names_out()
    n_num = len(NUMERICAL_FEATURES)
    n_cat = len(cat_features)
    bin_block = X_tr[:, n_num + n_cat:]
    unique_vals = set(np.unique(bin_block))
    assert unique_vals.issubset({0, 1, 0.0, 1.0}), \
        f"Binary columns contain unexpected values: {unique_vals}"


def test_unknown_category_handled(transformed, splits):
    """Unseen categories at inference time should not raise an error."""
    pp, *_ = transformed
    X_train, *_ = splits
    test_row = X_train.iloc[[0]].copy()
    test_row['Contract'] = 'Unknown-contract-type'
    try:
        result = pp.transform(test_row)
        expected_cols = pp.transform(X_train.iloc[[0]]).shape[1]
        assert result.shape[1] == expected_cols
    except Exception as e:
        pytest.fail(f"Unknown category raised an error: {e}")


def test_derived_features_present(splits):
    """Engineered features must exist in the dataframe."""
    X_train, *_ = splits
    assert 'charges_per_month' in X_train.columns
    assert 'num_services' in X_train.columns


def test_charges_per_month_non_negative(splits):
    """charges_per_month must be >= 0 for all rows."""
    X_train, X_val, X_test, *_ = splits
    for split in [X_train, X_val, X_test]:
        assert (split['charges_per_month'] >= 0).all(), \
            "charges_per_month has negative values"


def test_num_services_range(splits):
    """num_services must be between 0 and 6."""
    X_train, *_ = splits
    assert X_train['num_services'].between(0, 6).all(), \
        "num_services out of expected range 0-6"
    

# ── Edge case tests ───────────────────────────────────────────────────────────

def test_total_charges_null_fix(splits):
    """
    The 11 TotalCharges whitespace rows must be 0.0 after loading,
    not NaN. Verifies load_raw_data() handles the edge case correctly.
    """
    from src.features import load_raw_data
    df_raw = load_raw_data('data/raw/telco_churn.csv')
    assert df_raw['TotalCharges'].isnull().sum() == 0, \
        "TotalCharges still has NaN after load_raw_data()"
    assert df_raw['TotalCharges'].dtype == float, \
        "TotalCharges should be float after load_raw_data()"


def test_new_customer_charges_per_month(splits):
    """
    Customers with tenure=0 should have charges_per_month=0.0,
    not NaN or infinity. Verifies the +1 denominator protection.
    """
    X_train, X_val, X_test, *_ = splits
    all_X = pd.concat([X_train, X_val, X_test])
    new_customers = all_X[all_X['tenure'] == 0]

    if len(new_customers) > 0:
        assert new_customers['charges_per_month'].isnull().sum() == 0, \
            "tenure=0 customers have NaN in charges_per_month"
        assert not np.isinf(new_customers['charges_per_month']).any(), \
            "tenure=0 customers have Inf in charges_per_month"
        assert (new_customers['charges_per_month'] == 0.0).all(), \
            "tenure=0 customers should have charges_per_month=0.0"


def test_churn_binary_values(splits):
    """
    Target column must contain only 0 and 1 after clean_data().
    Catches a broken Yes/No mapping on the target.
    """
    _, _, _, y_train, y_val, y_test = splits
    for name, y in [('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        unique = set(y.unique())
        assert unique.issubset({0, 1}), \
            f"{name} contains values other than 0 and 1: {unique}"


def test_gender_binary_values(splits):
    """
    gender must be 0 or 1 after clean_data(), not 'Male'/'Female'.
    """
    X_train, *_ = splits
    unique = set(X_train['gender'].unique())
    assert unique.issubset({0, 1}), \
        f"gender contains unexpected values: {unique}"


def test_train_test_no_overlap(splits):
    """
    Train and test sets must share no rows.
    Catches a bug where the split produced duplicate indices.
    """
    X_train, _, X_test, *_ = splits
    train_idx = set(X_train.index)
    test_idx  = set(X_test.index)
    overlap   = train_idx.intersection(test_idx)
    assert len(overlap) == 0, \
        f"Train and test share {len(overlap)} row indices — potential leakage"


def test_train_val_no_overlap(splits):
    """
    Train and val sets must share no rows.
    """
    X_train, X_val, *_ = splits
    train_idx = set(X_train.index)
    val_idx   = set(X_val.index)
    overlap   = train_idx.intersection(val_idx)
    assert len(overlap) == 0, \
        f"Train and val share {len(overlap)} row indices — potential leakage"


def test_total_row_count(splits):
    """
    Train + val + test must add up to the full dataset (7043 rows).
    Catches a split that accidentally drops rows.
    """
    X_train, X_val, X_test, *_ = splits
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total == 7043, \
        f"Expected 7043 total rows, got {total} — rows were lost in splitting"


def test_preprocessor_is_unfitted():
    """
    build_preprocessor() must return an unfitted transformer each time.
    Catches accidental state sharing between calls.
    """
    from sklearn.exceptions import NotFittedError
    from src.features import build_preprocessor
    import numpy as np

    pp = build_preprocessor()
    dummy = pd.DataFrame({col: [0] for col in
                          ['tenure', 'MonthlyCharges', 'TotalCharges',
                           'charges_per_month', 'num_services',
                           'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies',
                           'Contract', 'PaymentMethod',
                           'gender', 'Partner', 'Dependents',
                           'PhoneService', 'PaperlessBilling',
                           'SeniorCitizen']})
    try:
        pp.transform(dummy)
        pytest.fail("Expected NotFittedError but transform succeeded")
    except NotFittedError:
        pass  # correct — transformer is unfitted


def test_num_services_counts_only_yes(splits):
    """
    num_services must count 'Yes' values only, not 'No internet service'.
    Verifies the lambda correctly filters to == 'Yes'.
    """
    from src.features import load_raw_data, clean_data, engineer_features
    df = load_raw_data('data/raw/telco_churn.csv')
    df = clean_data(df)
    df = engineer_features(df)

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']

    # Manual count for first 5 rows
    manual = (df[service_cols].iloc[:5] == 'Yes').sum(axis=1).values
    pipeline = df['num_services'].iloc[:5].values

    np.testing.assert_array_equal(
        manual, pipeline,
        err_msg="num_services doesn't match manual Yes count"
    )


def test_feature_columns_complete(splits):
    """
    X_train must contain all expected feature columns and nothing extra.
    Catches columns being accidentally added or dropped.
    """
    from src.features import (NUMERICAL_FEATURES, BINARY_FEATURES,
                               MULTI_CAT_FEATURES)
    X_train, *_ = splits
    expected = set(NUMERICAL_FEATURES + BINARY_FEATURES + MULTI_CAT_FEATURES)
    actual   = set(X_train.columns)
    missing  = expected - actual
    extra    = actual - expected

    assert not missing, f"Expected columns missing from X_train: {missing}"
    assert not extra,   f"Unexpected columns in X_train: {extra}"

def test_run_pipeline_saves_to_disk(tmp_path):
    """
    run_pipeline() with processed_dir should save 6 CSV files to disk.
    Uses pytest's tmp_path fixture for a clean temporary directory.
    """
    from src.features import run_pipeline

    run_pipeline(
        'data/raw/telco_churn.csv',
        processed_dir=str(tmp_path)
    )

    expected_files = ['X_train.csv', 'X_val.csv', 'X_test.csv',
                      'y_train.csv', 'y_val.csv', 'y_test.csv']

    for fname in expected_files:
        fpath = tmp_path / fname
        assert fpath.exists(), f"{fname} was not saved to disk"
        df = pd.read_csv(fpath)
        assert len(df) > 0, f"{fname} is empty"
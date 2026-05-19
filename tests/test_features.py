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
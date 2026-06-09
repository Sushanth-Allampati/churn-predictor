# tests/conftest.py
"""
Pytest configuration and fixtures shared across test modules.

For CI environments where model artifacts are not available,
we create a minimal mock model so API tests can run.
"""

import os
import pickle
import json
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def create_ci_model():
    """
    Create a minimal pipeline for CI testing.
    Not a real churn model — just enough to make the API start.
    """
    from src.features import (
        build_preprocessor, run_pipeline, NUMERICAL_FEATURES,
        MULTI_CAT_FEATURES, BINARY_FEATURES
    )

    # Use synthetic data if real data not available
    data_path = 'data/raw/telco_churn.csv'
    if not os.path.exists(data_path):
        from tests.create_test_data import create_minimal_dataset
        create_minimal_dataset(data_path)

    X_train, X_val, X_test, y_train, y_val, y_test = run_pipeline(data_path)

    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', LogisticRegression(
            max_iter=100, random_state=42,
            class_weight='balanced', solver='saga'
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def setup_ci_artifacts():
    """Create model artifacts needed for API tests and Docker build in CI."""

    os.makedirs('models', exist_ok=True)

    # Create pipeline.pkl if missing
    if not os.path.exists('models/pipeline.pkl'):
        print("Creating CI model pipeline...")
        pipeline = create_ci_model()
        with open('models/pipeline.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        print("  models/pipeline.pkl created")

    # Create threshold_config.json if missing
    if not os.path.exists('models/threshold_config.json'):
        config = {
            'optimal_threshold'  : 0.5,
            'threshold_metric'   : 'f1',
            'val_f1_at_threshold': 0.6,
            'val_precision'      : 0.6,
            'val_recall'         : 0.6,
            'calibration_method' : 'none',
            'base_brier_score'   : 0.2,
            'final_brier_score'  : 0.2,
            'note'               : 'CI test config',
        }
        with open('models/threshold_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("  models/threshold_config.json created")

    # Create test_metrics.json if missing
    if not os.path.exists('models/test_metrics.json'):
        metrics = {
            'roc_auc'  : 0.75,
            'pr_auc'   : 0.55,
            'f1'       : 0.55,
            'precision': 0.60,
            'recall'   : 0.55,
            'brier'    : 0.20,
        }
        with open('models/test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("  models/test_metrics.json created")

    # Create mlflow.db if missing
    if not os.path.exists('mlflow.db'):
        open('mlflow.db', 'w').close()
        print("  mlflow.db created (empty)")

    print("setup_ci_artifacts() complete")


if __name__ == '__main__':
    setup_ci_artifacts()


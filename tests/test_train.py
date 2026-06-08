"""
tests/test_train.py
───────────────────
Tests for src/train.py training functions.

These tests are intentionally lightweight — they verify the function
contracts without running full training loops (which would be slow in CI).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ── Import functions under test ───────────────────────────────────────────────

from src.train import (
    evaluate_model,
    _build_lr,
    _build_xgb,
    _build_lgbm,
    load_best_params,
    SCALE_POS_WEIGHT,
)
from src.features import run_pipeline, build_preprocessor

RAW_PATH = 'data/raw/telco_churn.csv'


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def splits():
    """Load data splits once for the whole test module."""
    return run_pipeline(RAW_PATH)


@pytest.fixture(scope='module')
def fitted_lr_pipeline(splits):
    """Train a small LR pipeline once for evaluation tests."""
    X_train, _, _, y_train, _, _ = splits
    model, params = _build_lr()
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model',        model),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ── SCALE_POS_WEIGHT tests ────────────────────────────────────────────────────

def test_scale_pos_weight_value():
    """scale_pos_weight must be positive and approximately 2.77."""
    assert SCALE_POS_WEIGHT > 0, "scale_pos_weight must be positive"
    assert 2.5 < SCALE_POS_WEIGHT < 3.0, \
        f"Expected ≈2.77 for 26.5% churn rate, got {SCALE_POS_WEIGHT:.4f}"


# ── Model builder tests ───────────────────────────────────────────────────────

def test_build_lr_returns_correct_type():
    model, params = _build_lr()
    assert isinstance(model, LogisticRegression)
    assert isinstance(params, dict)
    assert params['solver'] == 'saga', \
        "Must use saga solver for sklearn 1.5+ compatibility"
    assert params['class_weight'] == 'balanced', \
        "Must use balanced class weight for imbalanced data"


def test_build_xgb_returns_correct_type():
    model, params = _build_xgb()
    assert isinstance(model, XGBClassifier)
    assert isinstance(params, dict)
    assert params['scale_pos_weight'] > 1.0, \
        "scale_pos_weight must be > 1 for imbalanced data"
    assert params['eval_metric'] == 'aucpr', \
        "PR-AUC is the correct metric for imbalanced classification"


def test_build_lgbm_returns_correct_type():
    model, params = _build_lgbm()
    assert isinstance(model, LGBMClassifier)
    assert isinstance(params, dict)
    assert params['is_unbalance'] is True, \
        "is_unbalance must be True for class imbalance handling"


def test_all_builders_include_random_state():
    """Reproducibility requires random_state in all model configs."""
    for builder in [_build_lr, _build_xgb, _build_lgbm]:
        model, params = builder()
        assert model.random_state == 42, \
            f"{builder.__name__} model missing random_state=42"


# ── evaluate_model tests ──────────────────────────────────────────────────────

def test_evaluate_model_returns_correct_keys(fitted_lr_pipeline, splits):
    """evaluate_model must return all 6 expected metric keys."""
    _, X_val, _, _, y_val, _ = splits
    metrics = evaluate_model(fitted_lr_pipeline, X_val, y_val, 'val')

    expected_keys = [
        'val_accuracy', 'val_roc_auc', 'val_pr_auc',
        'val_f1', 'val_precision', 'val_recall',
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_evaluate_model_values_in_range(fitted_lr_pipeline, splits):
    """All metric values must be between 0 and 1."""
    _, X_val, _, _, y_val, _ = splits
    metrics = evaluate_model(fitted_lr_pipeline, X_val, y_val, 'val')

    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0, \
            f"Metric {key}={val:.4f} is outside [0, 1]"


def test_evaluate_model_prefix(fitted_lr_pipeline, splits):
    """Metric keys must use the provided split_name as prefix."""
    _, X_val, _, _, y_val, _ = splits

    for split_name in ['train', 'val', 'test']:
        metrics = evaluate_model(fitted_lr_pipeline, X_val, y_val, split_name)
        for key in metrics:
            assert key.startswith(split_name), \
                f"Key '{key}' doesn't start with '{split_name}'"


def test_evaluate_model_roc_auc_beats_random(fitted_lr_pipeline, splits):
    """
    A trained model should beat random on real data.
    Skipped in CI where synthetic random data produces near-random ROC-AUC.
    """
    _, X_val, _, _, y_val, _ = splits

    # Skip this check if using synthetic CI data (200 rows)
    if len(X_val) < 100:
        pytest.skip("Skipping ROC-AUC check on synthetic CI data")

    metrics = evaluate_model(fitted_lr_pipeline, X_val, y_val, 'val')
    assert metrics['val_roc_auc'] > 0.6, \
        f"ROC-AUC={metrics['val_roc_auc']:.4f} is unexpectedly low"


# ── load_best_params tests ────────────────────────────────────────────────────

def test_load_best_params_raises_for_missing_file():
    """FileNotFoundError must be raised if params file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_best_params('nonexistent_model')


def test_load_best_params_structure(tmp_path):
    """load_best_params must return dict with expected keys."""
    import json as json_module

    fake_params = {
        'model'          : 'LGBMClassifier',
        'best_params'    : {'n_estimators': 200, 'learning_rate': 0.05},
        'best_val_pr_auc': 0.72,
        'n_trials'       : 30,
    }

    # Write a real temp file and point load_best_params at it
    # by temporarily monkeypatching the path constant in src.train
    params_file = tmp_path / 'best_params_lgbm.json'
    params_file.write_text(json_module.dumps(fake_params))

    import src.train as train_module

    # Patch the path that load_best_params constructs internally
    original_exists = os.path.exists

    def fake_exists(path):
        if 'best_params_lgbm' in str(path):
            return True
        return original_exists(path)

    with patch('os.path.exists', side_effect=fake_exists):
        with patch('src.train.open',
                   create=True,
                   side_effect=lambda path, *a, **kw: open(str(params_file), *a, **kw)):
            result = load_best_params('lgbm')

    assert 'best_params' in result
    assert 'best_val_pr_auc' in result
    assert 'n_trials' in result


# ── Pipeline structure tests ──────────────────────────────────────────────────

def test_pipeline_has_correct_steps():
    """Pipeline must have exactly preprocessor and model steps."""
    model, _ = _build_lgbm()
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model',        model),
    ])
    assert list(pipeline.named_steps.keys()) == ['preprocessor', 'model']


def test_pipeline_predict_proba_shape(fitted_lr_pipeline, splits):
    """predict_proba must return shape (n_samples, 2) for binary classification."""
    _, X_val, _, _, y_val, _ = splits
    proba = fitted_lr_pipeline.predict_proba(X_val)
    assert proba.shape == (len(X_val), 2), \
        f"Expected shape ({len(X_val)}, 2), got {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0), \
        "Probabilities must sum to 1 for each sample"


def test_pipeline_probabilities_in_range(fitted_lr_pipeline, splits):
    """All predicted probabilities must be in [0, 1]."""
    _, X_val, _, _, y_val, _ = splits
    proba = fitted_lr_pipeline.predict_proba(X_val)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all(), \
        "Probabilities outside [0, 1] detected"
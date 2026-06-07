"""
api/model.py
────────────
Model loading for the churn prediction API.
Loads from pickle file for reliable deployment.
"""

import json
import os
import pickle

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_URI        = f'sqlite:///{os.path.join(_REPO_ROOT, "mlflow.db")}'
MODEL_NAME        = 'churn-model'
THRESHOLD_CONFIG  = os.path.join(_REPO_ROOT, 'models', 'threshold_config.json')
TEST_METRICS_PATH = os.path.join(_REPO_ROOT, 'models', 'test_metrics.json')
PIPELINE_PATH     = os.path.join(_REPO_ROOT, 'models', 'pipeline.pkl')

# ── Global state ──────────────────────────────────────────────────────────────

_state = {
    'pipeline'        : None,
    'threshold_config': None,
    'test_metrics'    : None,
    'model_version'   : None,
}


def load_model():
    """Load pipeline from pickle and config files."""

    print(f"load_model() called")
    print(f"  REPO ROOT     : {_REPO_ROOT}")
    print(f"  PIPELINE PATH : {PIPELINE_PATH}")
    print(f"  EXISTS        : {os.path.exists(PIPELINE_PATH)}")

    # ── Load threshold config ─────────────────────────────────────────────────
    with open(THRESHOLD_CONFIG) as f:
        _state['threshold_config'] = json.load(f)
    print("  threshold_config loaded OK")

    # ── Load test metrics ─────────────────────────────────────────────────────
    with open(TEST_METRICS_PATH) as f:
        _state['test_metrics'] = json.load(f)
    print("  test_metrics loaded OK")

    # ── Load pipeline from pickle ─────────────────────────────────────────────
    with open(PIPELINE_PATH, 'rb') as f:
        _state['pipeline'] = pickle.load(f)
    _state['model_version'] = 'v1'
    print("  pipeline.pkl loaded OK")
    print("load_model() complete")


def is_ready() -> bool:
    return _state['pipeline'] is not None


def get_pipeline():
    if _state['pipeline'] is None:
        raise RuntimeError("Model not loaded.")
    return _state['pipeline']


def get_threshold() -> float:
    tc = _state['threshold_config']
    if tc is None:
        return 0.5
    return tc.get('optimal_threshold', 0.5)


def __getattr__(name):
    if name in ('pipeline', 'threshold_config', 'test_metrics', 'model_version'):
        return _state.get(name)
    raise AttributeError(f"module 'api.model' has no attribute '{name}'")
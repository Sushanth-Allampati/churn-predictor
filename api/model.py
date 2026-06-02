"""
api/model.py
────────────
Model loading and prediction logic for the churn prediction API.
"""

import json
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Paths — absolute so they work regardless of working directory ──────────────
_REPO_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_URI        = f'sqlite:///{os.path.join(_REPO_ROOT, "mlflow.db")}'
MODEL_NAME        = 'churn-model'
THRESHOLD_CONFIG  = os.path.join(_REPO_ROOT, 'models', 'threshold_config.json')
TEST_METRICS_PATH = os.path.join(_REPO_ROOT, 'models', 'test_metrics.json')

# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    'pipeline'        : None,
    'threshold_config': None,
    'test_metrics'    : None,
    'model_version'   : None,
    'error'           : None,
}


def load_model():
    """Load model and config files. Populates _state dict."""

    print(f"load_model() called")
    print(f"  REPO ROOT : {_REPO_ROOT}")
    print(f"  MLFLOW URI: {MLFLOW_URI}")
    print(f"  THRESHOLD : {THRESHOLD_CONFIG}")
    print(f"  EXISTS    : {os.path.exists(THRESHOLD_CONFIG)}")

    # ── Load threshold config ─────────────────────────────────────────────────
    with open(THRESHOLD_CONFIG) as f:
        _state['threshold_config'] = json.load(f)
    print(f"  threshold_config loaded OK")

    # ── Load test metrics ─────────────────────────────────────────────────────
    with open(TEST_METRICS_PATH) as f:
        _state['test_metrics'] = json.load(f)
    print(f"  test_metrics loaded OK")

    # ── Load model from MLflow ────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No versions found for '{MODEL_NAME}'")

    versions_sorted = sorted(versions,
                             key=lambda v: int(v.version),
                             reverse=True)

    for v in versions_sorted:
        try:
            uri = f'runs:/{v.run_id}/pipeline'
            _state['pipeline']      = mlflow.sklearn.load_model(uri)
            _state['model_version'] = v.version
            print(f"  Model loaded: version={v.version} run_id={v.run_id[:8]}")
            break
        except Exception as e:
            print(f"  Skipping version {v.version}: {e}")
            continue

    if _state['pipeline'] is None:
        raise RuntimeError("No loadable model version found.")

    print("load_model() complete — all OK")


def is_ready() -> bool:
    return _state['pipeline'] is not None


def get_pipeline():
    return _state['pipeline']


def get_threshold() -> float:
    tc = _state['threshold_config']
    if tc is None:
        return 0.5
    return tc.get('optimal_threshold', 0.5)


# ── Expose as module attributes for backward compatibility ────────────────────
@property
def pipeline(self):
    return _state['pipeline']


# Direct attribute access used in health.py and routers
def _get(key):
    return _state.get(key)


# Make threshold_config and test_metrics and model_version accessible
# as module-level attributes via a module __getattr__
def __getattr__(name):
    if name in ('pipeline', 'threshold_config', 'test_metrics', 'model_version'):
        return _state.get(name)
    raise AttributeError(f"module 'api.model' has no attribute '{name}'")
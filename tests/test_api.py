"""
tests/test_api.py
─────────────────
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# ── Sample customer data ──────────────────────────────────────────────────────

VALID_CUSTOMER = {
    'gender'          : 'Male',
    'SeniorCitizen'   : 0,
    'Partner'         : 'Yes',
    'Dependents'      : 'No',
    'tenure'          : 2,
    'Contract'        : 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod'   : 'Electronic check',
    'MonthlyCharges'  : 85.50,
    'TotalCharges'    : 171.00,
    'PhoneService'    : 'Yes',
    'MultipleLines'   : 'No',
    'InternetService' : 'Fiber optic',
    'OnlineSecurity'  : 'No',
    'OnlineBackup'    : 'No',
    'DeviceProtection': 'No',
    'TechSupport'     : 'No',
    'StreamingTV'     : 'No',
    'StreamingMovies' : 'No',
}

LOYAL_CUSTOMER = {
    'gender'          : 'Female',
    'SeniorCitizen'   : 0,
    'Partner'         : 'Yes',
    'Dependents'      : 'Yes',
    'tenure'          : 60,
    'Contract'        : 'Two year',
    'PaperlessBilling': 'No',
    'PaymentMethod'   : 'Bank transfer (automatic)',
    'MonthlyCharges'  : 45.00,
    'TotalCharges'    : 2700.00,
    'PhoneService'    : 'Yes',
    'MultipleLines'   : 'Yes',
    'InternetService' : 'DSL',
    'OnlineSecurity'  : 'Yes',
    'OnlineBackup'    : 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport'     : 'Yes',
    'StreamingTV'     : 'No',
    'StreamingMovies' : 'No',
}


# ── Client fixture — lifespan runs correctly inside fixture ───────────────────

@pytest.fixture(scope='module')
def client():
    """
    Create TestClient inside a fixture so FastAPI lifespan
    (startup/shutdown) runs correctly.

    scope='module' means the client is created once per test file
    and shared across all tests — model loads only once.
    """
    with TestClient(app) as c:
        yield c


# ── Health endpoint tests ─────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get('/health')
    assert response.status_code == 200


def test_health_response_structure(client):
    response = client.get('/health')
    data = response.json()
    assert 'status' in data
    assert 'model_loaded' in data
    assert 'version' in data


def test_health_status_is_ok(client):
    response = client.get('/health')
    assert response.json()['status'] == 'ok'


def test_health_model_is_loaded(client):
    response = client.get('/health')
    assert response.json()['model_loaded'] is True


# ── Model info endpoint tests ─────────────────────────────────────────────────

def test_model_info_returns_200(client):
    response = client.get('/model-info')
    assert response.status_code == 200


def test_model_info_response_structure(client):
    response = client.get('/model-info')
    data = response.json()
    required_keys = [
        'model_name', 'model_version', 'optimal_threshold',
        'test_roc_auc', 'test_pr_auc', 'test_f1',
        'test_precision', 'test_recall', 'brier_score',
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"


def test_model_info_threshold_is_valid(client):
    response = client.get('/model-info')
    threshold = response.json()['optimal_threshold']
    assert 0.0 < threshold < 1.0, \
        f"Threshold {threshold} is outside (0, 1)"


def test_model_info_metrics_in_range(client):
    response = client.get('/model-info')
    data = response.json()
    for metric in ['test_roc_auc', 'test_pr_auc', 'test_f1',
                   'test_precision', 'test_recall']:
        val = data[metric]
        assert 0.0 <= val <= 1.0, \
            f"Metric {metric}={val} is outside [0, 1]"


def test_model_info_roc_auc_is_reasonable(client):
    """ROC-AUC should be well above 0.5 for a trained model."""
    response = client.get('/model-info')
    roc_auc = response.json()['test_roc_auc']
    assert roc_auc > 0.7, \
        f"test_roc_auc={roc_auc} is unexpectedly low"


# ── Root redirect test ────────────────────────────────────────────────────────

def test_root_redirects_to_docs(client):
    response = client.get('/', follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert '/docs' in response.headers.get('location', '')


# ── Schema validation tests ───────────────────────────────────────────────────

def test_invalid_gender_returns_422(client):
    """Invalid gender value should return 422 Unprocessable Entity."""
    invalid = {**VALID_CUSTOMER, 'gender': 'Unknown'}
    response = client.post('/predict', json=invalid)
    assert response.status_code in (404, 422)


def test_negative_tenure_returns_422(client):
    """Negative tenure should fail field validation."""
    invalid = {**VALID_CUSTOMER, 'tenure': -1}
    response = client.post('/predict', json=invalid)
    assert response.status_code in (404, 422)


def test_invalid_contract_returns_422(client):
    """Unknown contract type should fail field validation."""
    invalid = {**VALID_CUSTOMER, 'Contract': 'Week-to-week'}
    response = client.post('/predict', json=invalid)
    assert response.status_code in (404, 422)
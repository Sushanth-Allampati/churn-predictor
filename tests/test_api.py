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

# ── Predict endpoint tests ────────────────────────────────────────────────────

def test_predict_returns_200(client):
    response = client.post('/predict', json=VALID_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_structure(client):
    response = client.post('/predict', json=VALID_CUSTOMER)
    data = response.json()
    assert 'churn_probability' in data
    assert 'prediction'        in data
    assert 'risk_tier'         in data
    assert 'threshold_used'    in data


def test_predict_probability_in_range(client):
    response = client.post('/predict', json=VALID_CUSTOMER)
    prob = response.json()['churn_probability']
    assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]"


def test_predict_binary_prediction(client):
    """Prediction must be exactly 0 or 1."""
    response = client.post('/predict', json=VALID_CUSTOMER)
    pred = response.json()['prediction']
    assert pred in (0, 1), f"Prediction {pred} is not 0 or 1"


def test_predict_risk_tier_valid(client):
    """Risk tier must be High, Medium, or Low."""
    response = client.post('/predict', json=VALID_CUSTOMER)
    tier = response.json()['risk_tier']
    assert tier in ('High', 'Medium', 'Low'), \
        f"Risk tier '{tier}' is not valid"


def test_predict_high_risk_customer(client):
    """High-risk customer (short tenure, fiber, M2M) should score high."""
    response = client.post('/predict', json=VALID_CUSTOMER)
    data = response.json()
    # High-risk profile — probability should be meaningfully above 0.5
    assert data['churn_probability'] > 0.4, \
        f"Expected high probability for high-risk customer, got {data['churn_probability']}"


def test_predict_low_risk_customer(client):
    """Low-risk customer (long tenure, two-year contract) should score low."""
    response = client.post('/predict', json=LOYAL_CUSTOMER)
    data = response.json()
    assert data['churn_probability'] < 0.5, \
        f"Expected low probability for loyal customer, got {data['churn_probability']}"
    assert data['prediction'] == 0
    assert data['risk_tier'] == 'Low'


def test_predict_threshold_consistency(client):
    """
    Prediction must be consistent with probability and threshold.
    If prob >= threshold → prediction=1, else prediction=0.
    """
    response = client.post('/predict', json=VALID_CUSTOMER)
    data     = response.json()
    prob      = data['churn_probability']
    pred      = data['prediction']
    threshold = data['threshold_used']

    expected_pred = 1 if prob >= threshold else 0
    assert pred == expected_pred, \
        f"Prediction {pred} inconsistent with prob={prob} and threshold={threshold}"


def test_predict_invalid_gender_422(client):
    invalid = {**VALID_CUSTOMER, 'gender': 'Unknown'}
    response = client.post('/predict', json=invalid)
    assert response.status_code == 422


def test_predict_negative_tenure_422(client):
    invalid = {**VALID_CUSTOMER, 'tenure': -5}
    response = client.post('/predict', json=invalid)
    assert response.status_code == 422


def test_predict_invalid_contract_422(client):
    invalid = {**VALID_CUSTOMER, 'Contract': 'Daily'}
    response = client.post('/predict', json=invalid)
    assert response.status_code == 422


def test_predict_zero_monthly_charges_422(client):
    """MonthlyCharges must be > 0."""
    invalid = {**VALID_CUSTOMER, 'MonthlyCharges': 0.0}
    response = client.post('/predict', json=invalid)
    assert response.status_code == 422


def test_predict_missing_field_422(client):
    """Missing required field should return 422."""
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != 'tenure'}
    response = client.post('/predict', json=incomplete)
    assert response.status_code == 422


# ── Batch predict endpoint tests ──────────────────────────────────────────────

def test_batch_predict_returns_200(client):
    payload = {'customers': [VALID_CUSTOMER, LOYAL_CUSTOMER]}
    response = client.post('/predict/batch', json=payload)
    assert response.status_code == 200


def test_batch_predict_response_structure(client):
    payload = {'customers': [VALID_CUSTOMER, LOYAL_CUSTOMER]}
    response = client.post('/predict/batch', json=payload)
    data = response.json()
    assert 'predictions'  in data
    assert 'total'        in data
    assert 'high_risk'    in data
    assert 'medium_risk'  in data
    assert 'low_risk'     in data


def test_batch_predict_total_count(client):
    """Total count must equal number of customers sent."""
    payload = {'customers': [VALID_CUSTOMER, LOYAL_CUSTOMER]}
    response = client.post('/predict/batch', json=payload)
    data = response.json()
    assert data['total'] == 2
    assert len(data['predictions']) == 2


def test_batch_predict_risk_tier_sum(client):
    """high_risk + medium_risk + low_risk must equal total."""
    payload = {'customers': [VALID_CUSTOMER, LOYAL_CUSTOMER]}
    response = client.post('/predict/batch', json=payload)
    data = response.json()
    assert data['high_risk'] + data['medium_risk'] + data['low_risk'] == data['total']


def test_batch_predict_single_customer(client):
    """Batch with one customer should work the same as single predict."""
    single_resp = client.post('/predict', json=VALID_CUSTOMER)
    batch_resp  = client.post('/predict/batch',
                              json={'customers': [VALID_CUSTOMER]})

    single_prob = single_resp.json()['churn_probability']
    batch_prob  = batch_resp.json()['predictions'][0]['churn_probability']

    assert single_prob == batch_prob, \
        "Single and batch predictions differ for the same customer"


def test_batch_predict_empty_list_422(client):
    """Empty customer list should return 422."""
    payload = {'customers': []}
    response = client.post('/predict/batch', json=payload)
    assert response.status_code == 422


def test_batch_predict_invalid_customer_422(client):
    """Batch with one invalid customer should return 422."""
    invalid_customer = {**VALID_CUSTOMER, 'Contract': 'InvalidContract'}
    payload = {'customers': [VALID_CUSTOMER, invalid_customer]}
    response = client.post('/predict/batch', json=payload)
    assert response.status_code == 422

# ── Reason field tests ────────────────────────────────────────────────────────

def test_predict_response_has_reason(client):
    """Response must include a plain-English reason string."""
    response = client.post('/predict', json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert 'reason' in data
    assert isinstance(data['reason'], str)
    assert len(data['reason']) > 10, "Reason string is too short"


def test_predict_reason_is_different_for_different_customers(client):
    """Different customers should get different reason strings."""
    high_risk_resp = client.post('/predict', json=VALID_CUSTOMER)
    low_risk_resp  = client.post('/predict', json=LOYAL_CUSTOMER)

    high_reason = high_risk_resp.json()['reason']
    low_reason  = low_risk_resp.json()['reason']

    assert high_reason != low_reason, \
        "High-risk and low-risk customers got identical reasons"


# ── Explain endpoint tests ────────────────────────────────────────────────────

def test_explain_returns_200(client):
    response = client.post('/explain', json=VALID_CUSTOMER)
    assert response.status_code == 200


def test_explain_response_structure(client):
    response = client.post('/explain', json=VALID_CUSTOMER)
    data = response.json()
    required_keys = [
        'churn_probability', 'prediction', 'risk_tier',
        'threshold_used', 'reason', 'top_risk_factors',
        'top_protective', 'baseline_probability',
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"


def test_explain_feature_contributions_structure(client):
    """Each feature contribution must have feature, value, direction."""
    response = client.post('/explain', json=VALID_CUSTOMER)
    data = response.json()

    for contrib in data['top_risk_factors'] + data['top_protective']:
        assert 'feature'   in contrib
        assert 'value'     in contrib
        assert 'direction' in contrib
        assert contrib['direction'] in ('increases', 'decreases'), \
            f"Invalid direction: {contrib['direction']}"


def test_explain_baseline_is_reasonable(client):
    """Baseline probability should be close to the training churn rate."""
    response = client.post('/explain', json=VALID_CUSTOMER)
    baseline = response.json()['baseline_probability']
    assert 0.20 <= baseline <= 0.35, \
        f"Baseline {baseline} is far from expected churn rate ~0.265"


def test_explain_probability_matches_predict(client):
    """
    /explain and /predict must return the same probability
    for the same customer.
    """
    predict_resp = client.post('/predict',  json=VALID_CUSTOMER)
    explain_resp = client.post('/explain',  json=VALID_CUSTOMER)

    predict_prob = predict_resp.json()['churn_probability']
    explain_prob = explain_resp.json()['churn_probability']

    assert predict_prob == explain_prob, \
        f"/predict ({predict_prob}) and /explain ({explain_prob}) disagree"


def test_explain_invalid_input_422(client):
    """Invalid input to /explain should return 422."""
    invalid = {**VALID_CUSTOMER, 'tenure': -1}
    response = client.post('/explain', json=invalid)
    assert response.status_code == 422


# ── Logging middleware test ───────────────────────────────────────────────────

def test_request_completes_successfully(client):
    """
    Smoke test — verify the full request cycle completes without error.
    Tests that middleware doesn't interfere with normal requests.
    """
    for endpoint in ['/health', '/model-info']:
        response = client.get(endpoint)
        assert response.status_code == 200, \
            f"Endpoint {endpoint} failed with {response.status_code}"
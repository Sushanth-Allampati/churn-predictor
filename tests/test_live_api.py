"""
tests/test_live_api.py
──────────────────────
Live deployment tests — run against the actual deployed Render URL.

These tests require the API to be running at LIVE_URL.
They are NOT run in CI (too slow, requires network) but are run
manually before tagging a release.

Usage
-----
    pytest tests/test_live_api.py -v

Note: Free tier spins down after inactivity.
      First test may take 30-60s while the service wakes up.
"""

import time
import pytest
import urllib.request
import urllib.error
import json

# ── Configuration ─────────────────────────────────────────────────────────────

LIVE_URL = 'https://churn-predictor-api-gonj.onrender.com'
TIMEOUT  = 120   # seconds — generous for cold start


# ── Helper functions ──────────────────────────────────────────────────────────

def get(endpoint: str) -> dict:
    """Make a GET request to the live API."""
    url = f'{LIVE_URL}{endpoint}'
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def post(endpoint: str, data: dict) -> dict:
    """Make a POST request to the live API."""
    url  = f'{LIVE_URL}{endpoint}'
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def post_expecting_error(endpoint: str, data: dict) -> int:
    """Make a POST request and return the HTTP status code."""
    url  = f'{LIVE_URL}{endpoint}'
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


# ── Customer fixtures ─────────────────────────────────────────────────────────

HIGH_RISK = {
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 2, 'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.50, 'TotalCharges': 171.00,
    'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
    'OnlineBackup': 'No', 'DeviceProtection': 'No',
    'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
}

LOW_RISK = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'Yes', 'tenure': 60, 'Contract': 'Two year',
    'PaperlessBilling': 'No', 'PaymentMethod': 'Bank transfer (automatic)',
    'MonthlyCharges': 45.00, 'TotalCharges': 2700.00,
    'PhoneService': 'Yes', 'MultipleLines': 'Yes',
    'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
    'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes',
    'TechSupport': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Service availability
# ═════════════════════════════════════════════════════════════════════════════

class TestServiceAvailability:

    def test_service_is_reachable(self):
        """Service must respond — handles cold start with long timeout."""
        print(f"\nConnecting to {LIVE_URL}...")
        start = time.time()
        data  = get('/health')
        elapsed = time.time() - start
        print(f"  Response time: {elapsed:.1f}s")
        assert data['status'] == 'ok'

    def test_model_is_loaded(self):
        data = get('/health')
        assert data['model_loaded'] is True, \
            "Model failed to load on deployment"

    def test_service_version(self):
        data = get('/health')
        assert 'version' in data
        assert len(data['version']) > 0

    def test_docs_accessible(self):
        """Swagger UI must be accessible."""
        url = f'{LIVE_URL}/docs'
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            assert r.status == 200
            content = r.read().decode()
            assert 'swagger' in content.lower() or 'openapi' in content.lower()


# ═════════════════════════════════════════════════════════════════════════════
# 2. Model info endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestModelInfo:

    def test_returns_correct_fields(self):
        data = get('/model-info')
        for key in ['model_name', 'model_version', 'optimal_threshold',
                    'test_roc_auc', 'test_pr_auc', 'test_f1']:
            assert key in data, f"Missing: {key}"

    def test_metrics_are_reasonable(self):
        data = get('/model-info')
        assert data['test_roc_auc'] > 0.7, \
            f"ROC-AUC too low: {data['test_roc_auc']}"
        assert 0.0 < data['optimal_threshold'] < 1.0

    def test_model_name_correct(self):
        data = get('/model-info')
        assert data['model_name'] == 'churn-model'


# ═════════════════════════════════════════════════════════════════════════════
# 3. Prediction endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictions:

    def test_high_risk_prediction(self):
        data = post('/predict', HIGH_RISK)
        assert 'churn_probability' in data
        assert 'prediction'        in data
        assert 'risk_tier'         in data
        assert 'reason'            in data
        assert 0.0 <= data['churn_probability'] <= 1.0
        assert data['prediction'] in (0, 1)
        assert data['risk_tier'] in ('High', 'Medium', 'Low')

    def test_low_risk_prediction(self):
        data = post('/predict', LOW_RISK)
        assert data['churn_probability'] < 0.5
        assert data['prediction'] == 0
        assert data['risk_tier'] == 'Low'

    def test_high_risk_scores_higher_than_low_risk(self):
        high = post('/predict', HIGH_RISK)['churn_probability']
        low  = post('/predict', LOW_RISK)['churn_probability']
        assert high > low, \
            f"High risk ({high:.3f}) should be higher than low risk ({low:.3f})"

    def test_prediction_consistent_with_probability(self):
        data      = post('/predict', HIGH_RISK)
        prob      = data['churn_probability']
        pred      = data['prediction']
        threshold = data['threshold_used']
        expected  = 1 if prob >= threshold else 0
        assert pred == expected

    def test_reason_is_meaningful(self):
        data = post('/predict', HIGH_RISK)
        assert isinstance(data['reason'], str)
        assert len(data['reason']) >= 20

    def test_predictions_are_deterministic(self):
        """Same input must produce identical results on live server."""
        prob1 = post('/predict', HIGH_RISK)['churn_probability']
        prob2 = post('/predict', HIGH_RISK)['churn_probability']
        assert prob1 == prob2, \
            f"Non-deterministic: {prob1} vs {prob2}"

    def test_all_contract_types(self):
        for contract in ['Month-to-month', 'One year', 'Two year']:
            customer = {**HIGH_RISK, 'Contract': contract}
            data     = post('/predict', customer)
            assert 0.0 <= data['churn_probability'] <= 1.0

    def test_edge_case_new_customer(self):
        """Brand new customer with tenure=0 and TotalCharges=0."""
        new_customer = {
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No',
            'Dependents': 'No', 'tenure': 0, 'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 0.01, 'TotalCharges': 0.0,
            'PhoneService': 'No', 'MultipleLines': 'No phone service',
            'InternetService': 'No', 'OnlineSecurity': 'No internet service',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No internet service',
            'TechSupport': 'No internet service',
            'StreamingTV': 'No internet service',
            'StreamingMovies': 'No internet service',
        }
        data = post('/predict', new_customer)
        assert 0.0 <= data['churn_probability'] <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 4. Validation on live server
# ═════════════════════════════════════════════════════════════════════════════

class TestLiveValidation:

    def test_invalid_gender_returns_422(self):
        status = post_expecting_error(
            '/predict', {**HIGH_RISK, 'gender': 'Unknown'}
        )
        assert status == 422

    def test_negative_tenure_returns_422(self):
        status = post_expecting_error(
            '/predict', {**HIGH_RISK, 'tenure': -1}
        )
        assert status == 422

    def test_invalid_contract_returns_422(self):
        status = post_expecting_error(
            '/predict', {**HIGH_RISK, 'Contract': 'Weekly'}
        )
        assert status == 422

    def test_missing_field_returns_422(self):
        incomplete = {k: v for k, v in HIGH_RISK.items() if k != 'tenure'}
        status = post_expecting_error('/predict', incomplete)
        assert status == 422


# ═════════════════════════════════════════════════════════════════════════════
# 5. Batch endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestBatchLive:

    def test_batch_two_customers(self):
        payload = {'customers': [HIGH_RISK, LOW_RISK]}
        data    = post('/predict/batch', payload)
        assert data['total'] == 2
        assert len(data['predictions']) == 2
        assert data['high_risk'] + data['medium_risk'] + data['low_risk'] == 2

    def test_batch_matches_single(self):
        """Batch result must match single predict for same customer."""
        single = post('/predict', HIGH_RISK)['churn_probability']
        batch  = post('/predict/batch',
                      {'customers': [HIGH_RISK]})['predictions'][0]['churn_probability']
        assert single == batch


# ═════════════════════════════════════════════════════════════════════════════
# 6. Explain endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestExplainLive:

    def test_explain_returns_correct_structure(self):
        data = post('/explain', HIGH_RISK)
        for key in ['churn_probability', 'prediction', 'risk_tier',
                    'reason', 'top_risk_factors', 'top_protective',
                    'baseline_probability']:
            assert key in data, f"Missing: {key}"

    def test_explain_matches_predict(self):
        predict_prob = post('/predict', HIGH_RISK)['churn_probability']
        explain_prob = post('/explain', HIGH_RISK)['churn_probability']
        assert predict_prob == explain_prob


# ═════════════════════════════════════════════════════════════════════════════
# 7. Performance on live server
# ═════════════════════════════════════════════════════════════════════════════

class TestLivePerformance:

    def test_warm_prediction_under_2s(self):
        """
        After the service is warm (previous tests hit it),
        a single prediction should complete under 2 seconds.
        The 2s threshold accounts for network latency to Singapore.
        """
        # Warm up with one request
        post('/predict', HIGH_RISK)

        # Now measure
        start   = time.time()
        data    = post('/predict', HIGH_RISK)
        elapsed = time.time() - start

        assert data['prediction'] in (0, 1)
        assert elapsed < 2.0, \
            f"Warm prediction took {elapsed:.2f}s — exceeds 2s threshold"
        print(f"\n  Warm prediction latency: {elapsed*1000:.0f}ms")

    def test_five_sequential_requests(self):
        """Five sequential requests must all succeed."""
        results = []
        for i in range(5):
            data = post('/predict', HIGH_RISK)
            results.append(data['churn_probability'])

        assert len(results) == 5
        assert all(0.0 <= p <= 1.0 for p in results)
        # All should be identical (deterministic)
        assert len(set(results)) == 1, \
            f"Non-deterministic results: {results}"
        print(f"\n  All 5 predictions: {results[0]:.4f}")
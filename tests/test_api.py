"""
tests/test_api.py
─────────────────
Comprehensive API test suite for the Churn Prediction API.

Test categories
---------------
1. Health & model-info endpoints
2. Schema validation (422 responses)
3. Predict endpoint — happy path
4. Predict endpoint — behaviour correctness
5. Predict endpoint — edge cases
6. Batch predict endpoint
7. Explain endpoint
8. Cross-endpoint consistency
9. Performance smoke test
"""

import time
import pytest
from fastapi.testclient import TestClient
from api.main import app


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def client():
    """
    Create TestClient using context manager so FastAPI lifespan
    (startup/shutdown) runs correctly. scope='module' means the
    model loads once and is shared across all tests.
    """
    with TestClient(app) as c:
        yield c


# ── Customer fixtures ─────────────────────────────────────────────────────────

HIGH_RISK_CUSTOMER = {
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

LOW_RISK_CUSTOMER = {
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

NEW_CUSTOMER = {
    'gender'          : 'Male',
    'SeniorCitizen'   : 0,
    'Partner'         : 'No',
    'Dependents'      : 'No',
    'tenure'          : 0,
    'Contract'        : 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod'   : 'Electronic check',
    'MonthlyCharges'  : 0.01,    # minimum valid value
    'TotalCharges'    : 0.0,     # new customer — no charges yet
    'PhoneService'    : 'No',
    'MultipleLines'   : 'No phone service',
    'InternetService' : 'No',
    'OnlineSecurity'  : 'No internet service',
    'OnlineBackup'    : 'No internet service',
    'DeviceProtection': 'No internet service',
    'TechSupport'     : 'No internet service',
    'StreamingTV'     : 'No internet service',
    'StreamingMovies' : 'No internet service',
}

SENIOR_TWO_YEAR = {
    'gender'          : 'Female',
    'SeniorCitizen'   : 1,
    'Partner'         : 'No',
    'Dependents'      : 'No',
    'tenure'          : 24,
    'Contract'        : 'Two year',
    'PaperlessBilling': 'Yes',
    'PaymentMethod'   : 'Credit card (automatic)',
    'MonthlyCharges'  : 70.00,
    'TotalCharges'    : 1680.00,
    'PhoneService'    : 'Yes',
    'MultipleLines'   : 'Yes',
    'InternetService' : 'Fiber optic',
    'OnlineSecurity'  : 'Yes',
    'OnlineBackup'    : 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport'     : 'Yes',
    'StreamingTV'     : 'Yes',
    'StreamingMovies' : 'Yes',
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Health and model-info endpoints
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_returns_200(self, client):
        assert client.get('/health').status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.get('/health').json()
        assert 'status'      in data
        assert 'model_loaded' in data
        assert 'version'     in data

    def test_status_is_ok(self, client):
        assert client.get('/health').json()['status'] == 'ok'

    def test_model_is_loaded(self, client):
        assert client.get('/health').json()['model_loaded'] is True

    def test_version_is_string(self, client):
        version = client.get('/health').json()['version']
        assert isinstance(version, str)
        assert len(version) > 0


class TestModelInfoEndpoint:

    def test_returns_200(self, client):
        assert client.get('/model-info').status_code == 200

    def test_response_has_required_fields(self, client):
        data     = client.get('/model-info').json()
        required = [
            'model_name', 'model_version', 'optimal_threshold',
            'calibration', 'test_roc_auc', 'test_pr_auc', 'test_f1',
            'test_precision', 'test_recall', 'brier_score',
        ]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_threshold_is_valid(self, client):
        threshold = client.get('/model-info').json()['optimal_threshold']
        assert 0.0 < threshold < 1.0

    def test_metrics_in_valid_range(self, client):
        data = client.get('/model-info').json()
        for metric in ['test_roc_auc', 'test_pr_auc', 'test_f1',
                       'test_precision', 'test_recall', 'brier_score']:
            val = data[metric]
            assert 0.0 <= val <= 1.0, f"{metric}={val} outside [0,1]"

    def test_roc_auc_beats_random(self, client):
        roc = client.get('/model-info').json()['test_roc_auc']
        assert roc > 0.7, f"ROC-AUC={roc} is suspiciously low"

    def test_model_name_is_correct(self, client):
        name = client.get('/model-info').json()['model_name']
        assert name == 'churn-model'


# ═════════════════════════════════════════════════════════════════════════════
# 2. Schema validation — 422 responses
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:

    def _post(self, client, overrides):
        return client.post('/predict', json={**HIGH_RISK_CUSTOMER, **overrides})

    def test_invalid_gender_returns_422(self, client):
        assert self._post(client, {'gender': 'Unknown'}).status_code == 422

    def test_negative_tenure_returns_422(self, client):
        assert self._post(client, {'tenure': -1}).status_code == 422

    def test_tenure_above_max_returns_422(self, client):
        assert self._post(client, {'tenure': 999}).status_code == 422

    def test_zero_monthly_charges_returns_422(self, client):
        assert self._post(client, {'MonthlyCharges': 0.0}).status_code == 422

    def test_negative_monthly_charges_returns_422(self, client):
        assert self._post(client, {'MonthlyCharges': -10.0}).status_code == 422

    def test_negative_total_charges_returns_422(self, client):
        assert self._post(client, {'TotalCharges': -1.0}).status_code == 422

    def test_invalid_contract_returns_422(self, client):
        assert self._post(client, {'Contract': 'Daily'}).status_code == 422

    def test_invalid_internet_service_returns_422(self, client):
        assert self._post(client, {'InternetService': 'Satellite'}).status_code == 422

    def test_invalid_payment_method_returns_422(self, client):
        assert self._post(client, {'PaymentMethod': 'Bitcoin'}).status_code == 422

    def test_invalid_multiple_lines_returns_422(self, client):
        assert self._post(client, {'MultipleLines': 'Maybe'}).status_code == 422

    def test_invalid_online_security_returns_422(self, client):
        assert self._post(client,
                          {'OnlineSecurity': 'Sometimes'}).status_code == 422

    def test_senior_citizen_must_be_0_or_1(self, client):
        assert self._post(client, {'SeniorCitizen': 2}).status_code == 422

    def test_senior_citizen_negative_returns_422(self, client):
        assert self._post(client, {'SeniorCitizen': -1}).status_code == 422

    def test_missing_tenure_returns_422(self, client):
        incomplete = {k: v for k, v in HIGH_RISK_CUSTOMER.items()
                      if k != 'tenure'}
        assert client.post('/predict', json=incomplete).status_code == 422

    def test_missing_contract_returns_422(self, client):
        incomplete = {k: v for k, v in HIGH_RISK_CUSTOMER.items()
                      if k != 'Contract'}
        assert client.post('/predict', json=incomplete).status_code == 422

    def test_empty_body_returns_422(self, client):
        assert client.post('/predict', json={}).status_code == 422

    def test_422_response_has_detail(self, client):
        """422 responses must include a detail field explaining the error."""
        response = self._post(client, {'gender': 'Robot'})
        assert response.status_code == 422
        data = response.json()
        assert 'detail' in data, "422 response missing 'detail' field"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Predict endpoint — happy path
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictHappyPath:

    def test_returns_200(self, client):
        assert client.post('/predict',
                           json=HIGH_RISK_CUSTOMER).status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post('/predict', json=HIGH_RISK_CUSTOMER).json()
        for key in ['churn_probability', 'prediction', 'risk_tier',
                    'threshold_used', 'reason']:
            assert key in data, f"Missing field: {key}"

    def test_probability_in_range(self, client):
        prob = client.post('/predict',
                           json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        assert 0.0 <= prob <= 1.0

    def test_prediction_is_binary(self, client):
        pred = client.post('/predict',
                           json=HIGH_RISK_CUSTOMER).json()['prediction']
        assert pred in (0, 1)

    def test_risk_tier_is_valid(self, client):
        tier = client.post('/predict',
                           json=HIGH_RISK_CUSTOMER).json()['risk_tier']
        assert tier in ('High', 'Medium', 'Low')

    def test_threshold_matches_model_info(self, client):
        """Threshold in /predict must match /model-info."""
        predict_thresh = client.post(
            '/predict', json=HIGH_RISK_CUSTOMER
        ).json()['threshold_used']
        model_thresh = client.get('/model-info').json()['optimal_threshold']
        assert predict_thresh == model_thresh

    def test_reason_is_non_empty_string(self, client):
        reason = client.post('/predict',
                             json=HIGH_RISK_CUSTOMER).json()['reason']
        assert isinstance(reason, str)
        assert len(reason) >= 20

    def test_prediction_consistent_with_probability(self, client):
        """prediction=1 iff probability >= threshold."""
        data      = client.post('/predict', json=HIGH_RISK_CUSTOMER).json()
        prob      = data['churn_probability']
        pred      = data['prediction']
        threshold = data['threshold_used']
        expected  = 1 if prob >= threshold else 0
        assert pred == expected, \
            f"prediction={pred} inconsistent with prob={prob}, threshold={threshold}"

    def test_risk_tier_consistent_with_probability(self, client):
        """Risk tier must be consistent with probability and threshold."""
        data      = client.post('/predict', json=HIGH_RISK_CUSTOMER).json()
        prob      = data['churn_probability']
        tier      = data['risk_tier']
        threshold = data['threshold_used']

        if prob >= threshold + 0.15:
            assert tier == 'High',   f"Expected High, got {tier} (prob={prob})"
        elif prob >= threshold:
            assert tier == 'Medium', f"Expected Medium, got {tier} (prob={prob})"
        else:
            assert tier == 'Low',    f"Expected Low, got {tier} (prob={prob})"


# ═════════════════════════════════════════════════════════════════════════════
# 4. Predict endpoint — behaviour correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictBehaviour:

    def test_high_risk_customer_scores_high(self, client):
        """
        2-month tenure, fiber optic, month-to-month should score
        meaningfully higher than the baseline churn rate.
        """
        prob = client.post('/predict',
                           json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        assert prob > 0.4, \
            f"High-risk customer scored only {prob:.3f}"

    def test_low_risk_customer_scores_low(self, client):
        """
        60-month tenure, two-year contract, many services should score low.
        """
        data = client.post('/predict', json=LOW_RISK_CUSTOMER).json()
        assert data['churn_probability'] < 0.4, \
            f"Low-risk customer scored {data['churn_probability']:.3f}"
        assert data['prediction'] == 0
        assert data['risk_tier'] == 'Low'

    def test_high_risk_scores_higher_than_low_risk(self, client):
        """High-risk customer must always score higher than low-risk customer."""
        high_prob = client.post('/predict',
                                json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        low_prob  = client.post('/predict',
                                json=LOW_RISK_CUSTOMER).json()['churn_probability']
        assert high_prob > low_prob, \
            f"High-risk ({high_prob:.3f}) did not outscore low-risk ({low_prob:.3f})"

    def test_new_customer_with_no_services(self, client):
        """Brand-new customer with no services should be processable."""
        response = client.post('/predict', json=NEW_CUSTOMER)
        assert response.status_code == 200
        prob = response.json()['churn_probability']
        assert 0.0 <= prob <= 1.0

    def test_senior_with_two_year_contract(self, client):
        """Senior on two-year contract with all services should be low risk."""
        data = client.post('/predict', json=SENIOR_TWO_YEAR).json()
        assert data['churn_probability'] < 0.6, \
            f"Senior with two-year contract scored high: {data['churn_probability']:.3f}"

    def test_predictions_are_deterministic(self, client):
        """Same input must produce identical output every time."""
        prob1 = client.post('/predict',
                            json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        prob2 = client.post('/predict',
                            json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        assert prob1 == prob2, \
            f"Non-deterministic predictions: {prob1} vs {prob2}"

    def test_all_contract_types_accepted(self, client):
        """All three contract types must be accepted and produce valid responses."""
        for contract in ['Month-to-month', 'One year', 'Two year']:
            customer = {**HIGH_RISK_CUSTOMER, 'Contract': contract}
            response = client.post('/predict', json=customer)
            assert response.status_code == 200, \
                f"Contract '{contract}' returned {response.status_code}"
            prob = response.json()['churn_probability']
            assert 0.0 <= prob <= 1.0

    def test_contract_type_affects_prediction(self, client):
        """Two-year contract should produce lower probability than month-to-month."""
        mtm = client.post('/predict',
                          json={**HIGH_RISK_CUSTOMER,
                                'Contract': 'Month-to-month'}).json()['churn_probability']
        two_yr = client.post('/predict',
                             json={**HIGH_RISK_CUSTOMER,
                                   'Contract': 'Two year'}).json()['churn_probability']
        assert two_yr < mtm, \
            f"Two-year contract ({two_yr:.3f}) should score lower than M2M ({mtm:.3f})"

    def test_tenure_affects_prediction(self, client):
        """Longer tenure should produce lower churn probability."""
        short = client.post('/predict',
                            json={**HIGH_RISK_CUSTOMER,
                                  'tenure': 1,
                                  'TotalCharges': 85.5}).json()['churn_probability']
        long_ = client.post('/predict',
                            json={**HIGH_RISK_CUSTOMER,
                                  'tenure': 60,
                                  'TotalCharges': 5130.0}).json()['churn_probability']
        assert long_ < short, \
            f"Long tenure ({long_:.3f}) should score lower than short ({short:.3f})"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Predict endpoint — edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestPredictEdgeCases:

    def test_minimum_tenure(self, client):
        """Tenure of 0 (new customer) should work without errors."""
        customer = {**HIGH_RISK_CUSTOMER,
                    'tenure': 0, 'TotalCharges': 0.0}
        response = client.post('/predict', json=customer)
        assert response.status_code == 200

    def test_maximum_tenure(self, client):
        """Tenure of 120 (10 years) should work without errors."""
        customer = {**HIGH_RISK_CUSTOMER,
                    'tenure': 120, 'TotalCharges': 10260.0}
        response = client.post('/predict', json=customer)
        assert response.status_code == 200

    def test_minimum_monthly_charges(self, client):
        """Minimum valid MonthlyCharges (0.01) should work."""
        customer = {**HIGH_RISK_CUSTOMER, 'MonthlyCharges': 0.01}
        response = client.post('/predict', json=customer)
        assert response.status_code == 200

    def test_very_high_monthly_charges(self, client):
        """Very high monthly charges should work without errors."""
        customer = {**HIGH_RISK_CUSTOMER, 'MonthlyCharges': 500.0}
        response = client.post('/predict', json=customer)
        assert response.status_code == 200

    def test_no_internet_service_customer(self, client):
        """Customer with no internet service should work correctly."""
        response = client.post('/predict', json=NEW_CUSTOMER)
        assert response.status_code == 200
        assert 0.0 <= response.json()['churn_probability'] <= 1.0

    def test_all_services_enabled(self, client):
        """Customer with all add-on services should work correctly."""
        customer = {
            **HIGH_RISK_CUSTOMER,
            'OnlineSecurity'  : 'Yes',
            'OnlineBackup'    : 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport'     : 'Yes',
            'StreamingTV'     : 'Yes',
            'StreamingMovies' : 'Yes',
        }
        response = client.post('/predict', json=customer)
        assert response.status_code == 200

    def test_total_charges_zero_with_zero_tenure(self, client):
        """TotalCharges=0 with tenure=0 should not cause division by zero."""
        customer = {**NEW_CUSTOMER, 'tenure': 0, 'TotalCharges': 0.0}
        response = client.post('/predict', json=customer)
        assert response.status_code == 200

    def test_all_payment_methods_accepted(self, client):
        """All four payment methods must be accepted."""
        methods = [
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)',
        ]
        for method in methods:
            customer = {**HIGH_RISK_CUSTOMER, 'PaymentMethod': method}
            response = client.post('/predict', json=customer)
            assert response.status_code == 200, \
                f"PaymentMethod '{method}' returned {response.status_code}"

    def test_all_internet_service_types(self, client):
        """DSL, Fiber optic, and No must all be accepted."""
        for service in ['DSL', 'Fiber optic', 'No']:
            if service == 'No':
                customer = {**NEW_CUSTOMER}
            else:
                customer = {**HIGH_RISK_CUSTOMER, 'InternetService': service}
            response = client.post('/predict', json=customer)
            assert response.status_code == 200, \
                f"InternetService '{service}' returned {response.status_code}"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Batch predict endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestBatchPredict:

    def test_returns_200(self, client):
        payload  = {'customers': [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]}
        response = client.post('/predict/batch', json=payload)
        assert response.status_code == 200

    def test_response_structure(self, client):
        payload = {'customers': [HIGH_RISK_CUSTOMER]}
        data    = client.post('/predict/batch', json=payload).json()
        for key in ['predictions', 'total', 'high_risk',
                    'medium_risk', 'low_risk']:
            assert key in data, f"Missing field: {key}"

    def test_total_equals_input_count(self, client):
        customers = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER, NEW_CUSTOMER]
        data = client.post('/predict/batch',
                           json={'customers': customers}).json()
        assert data['total'] == 3
        assert len(data['predictions']) == 3

    def test_risk_tier_counts_sum_to_total(self, client):
        payload = {'customers': [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]}
        data    = client.post('/predict/batch', json=payload).json()
        assert (data['high_risk'] + data['medium_risk'] + data['low_risk']
                == data['total'])

    def test_single_customer_batch_matches_single_predict(self, client):
        """Batch of 1 must produce identical result to single /predict."""
        single = client.post('/predict',
                             json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        batch  = client.post('/predict/batch',
                             json={'customers': [HIGH_RISK_CUSTOMER]}
                             ).json()['predictions'][0]['churn_probability']
        assert single == batch, \
            f"Single={single} and batch={batch} disagree"

    def test_all_predictions_have_required_fields(self, client):
        payload = {'customers': [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]}
        data    = client.post('/predict/batch', json=payload).json()
        for pred in data['predictions']:
            for key in ['churn_probability', 'prediction',
                        'risk_tier', 'threshold_used', 'reason']:
                assert key in pred, f"Missing field in batch prediction: {key}"

    def test_empty_list_returns_422(self, client):
        assert client.post('/predict/batch',
                           json={'customers': []}).status_code == 422

    def test_invalid_customer_in_batch_returns_422(self, client):
        bad = {**HIGH_RISK_CUSTOMER, 'Contract': 'InvalidType'}
        payload = {'customers': [HIGH_RISK_CUSTOMER, bad]}
        assert client.post('/predict/batch', json=payload).status_code == 422

    def test_large_batch(self, client):
        """Batch of 50 customers should complete without errors."""
        customers = [HIGH_RISK_CUSTOMER] * 25 + [LOW_RISK_CUSTOMER] * 25
        data = client.post('/predict/batch',
                           json={'customers': customers}).json()
        assert data['total'] == 50
        assert len(data['predictions']) == 50


# ═════════════════════════════════════════════════════════════════════════════
# 7. Explain endpoint
# ═════════════════════════════════════════════════════════════════════════════

class TestExplainEndpoint:

    def test_returns_200(self, client):
        assert client.post('/explain',
                           json=HIGH_RISK_CUSTOMER).status_code == 200

    def test_response_structure(self, client):
        data = client.post('/explain', json=HIGH_RISK_CUSTOMER).json()
        for key in ['churn_probability', 'prediction', 'risk_tier',
                    'threshold_used', 'reason', 'top_risk_factors',
                    'top_protective', 'baseline_probability']:
            assert key in data, f"Missing field: {key}"

    def test_feature_contribution_structure(self, client):
        data = client.post('/explain', json=HIGH_RISK_CUSTOMER).json()
        for contrib in data['top_risk_factors'] + data['top_protective']:
            assert 'feature'   in contrib
            assert 'value'     in contrib
            assert 'direction' in contrib
            assert contrib['direction'] in ('increases', 'decreases')

    def test_baseline_near_churn_rate(self, client):
        baseline = client.post('/explain',
                               json=HIGH_RISK_CUSTOMER).json()['baseline_probability']
        assert 0.20 <= baseline <= 0.35, \
            f"Baseline {baseline} is far from expected ~0.265"

    def test_probability_matches_predict(self, client):
        """Probability from /explain must equal /predict for same customer."""
        predict_prob = client.post('/predict',
                                   json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        explain_prob = client.post('/explain',
                                   json=HIGH_RISK_CUSTOMER).json()['churn_probability']
        assert predict_prob == explain_prob

    def test_invalid_input_returns_422(self, client):
        invalid  = {**HIGH_RISK_CUSTOMER, 'tenure': -5}
        response = client.post('/explain', json=invalid)
        assert response.status_code == 422

    def test_lists_are_not_empty(self, client):
        data = client.post('/explain', json=HIGH_RISK_CUSTOMER).json()
        # At least one contribution should be present
        total = len(data['top_risk_factors']) + len(data['top_protective'])
        assert total > 0, "No feature contributions returned"


# ═════════════════════════════════════════════════════════════════════════════
# 8. Cross-endpoint consistency
# ═════════════════════════════════════════════════════════════════════════════

class TestCrossEndpointConsistency:

    def test_predict_and_explain_agree(self, client):
        """All shared fields in /predict and /explain must be identical."""
        predict = client.post('/predict', json=HIGH_RISK_CUSTOMER).json()
        explain = client.post('/explain', json=HIGH_RISK_CUSTOMER).json()

        for field in ['churn_probability', 'prediction',
                      'risk_tier', 'threshold_used']:
            assert predict[field] == explain[field], \
                f"Field '{field}' differs: predict={predict[field]}, explain={explain[field]}"

    def test_threshold_consistent_across_endpoints(self, client):
        """Threshold must be the same from /model-info, /predict, /explain."""
        model_thresh   = client.get('/model-info').json()['optimal_threshold']
        predict_thresh = client.post('/predict',
                                     json=HIGH_RISK_CUSTOMER).json()['threshold_used']
        explain_thresh = client.post('/explain',
                                     json=HIGH_RISK_CUSTOMER).json()['threshold_used']

        assert model_thresh   == predict_thresh, \
            "Threshold differs between /model-info and /predict"
        assert predict_thresh == explain_thresh, \
            "Threshold differs between /predict and /explain"

    def test_root_redirects_to_docs(self, client):
        response = client.get('/', follow_redirects=False)
        assert response.status_code in (301, 302, 307, 308)
        assert '/docs' in response.headers.get('location', '')


# ═════════════════════════════════════════════════════════════════════════════
# 9. Performance smoke test
# ═════════════════════════════════════════════════════════════════════════════

class TestPerformance:

    def test_single_predict_under_500ms(self, client):
        """Single prediction should complete in under 500ms."""
        start    = time.time()
        response = client.post('/predict', json=HIGH_RISK_CUSTOMER)
        elapsed  = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 500, \
            f"/predict took {elapsed:.0f}ms — exceeds 500ms threshold"

    def test_health_check_under_100ms(self, client):
        """Health check should be very fast — under 100ms."""
        start   = time.time()
        response = client.get('/health')
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 100, \
            f"/health took {elapsed:.0f}ms — exceeds 100ms threshold"

    def test_batch_50_customers_under_5s(self, client):
        """Batch of 50 customers should complete under 5 seconds."""
        customers = [HIGH_RISK_CUSTOMER] * 50
        start     = time.time()
        response  = client.post('/predict/batch',
                                json={'customers': customers})
        elapsed   = time.time() - start

        assert response.status_code == 200
        assert elapsed < 5.0, \
            f"Batch of 50 took {elapsed:.1f}s — exceeds 5s threshold"

    def test_ten_sequential_requests(self, client):
        """Ten sequential predictions should all succeed."""
        results = []
        for _ in range(10):
            response = client.post('/predict', json=HIGH_RISK_CUSTOMER)
            results.append(response.status_code)
        assert all(s == 200 for s in results), \
            f"Some requests failed: {results}"
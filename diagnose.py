# save as manual_check.py, delete after
import requests

BASE = 'http://localhost:8000'

def check(name, response, expected_status):
    status = response.status_code
    ok = '✓' if status == expected_status else '✗'
    print(f"{ok} {name}: {status}")
    if status != expected_status:
        print(f"  Expected {expected_status}, got {status}")
        print(f"  Body: {response.json()}")

VALID = {
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 2, 'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.50, 'TotalCharges': 171.00,
    'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
    'OnlineBackup': 'No', 'DeviceProtection': 'No',
    'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
}

print("Manual API verification")
print("="*45)

# Happy path
check("GET  /health",      requests.get(f'{BASE}/health'),          200)
check("GET  /model-info",  requests.get(f'{BASE}/model-info'),      200)
check("POST /predict",     requests.post(f'{BASE}/predict', json=VALID), 200)
check("POST /explain",     requests.post(f'{BASE}/explain', json=VALID), 200)
check("POST /predict/batch",
      requests.post(f'{BASE}/predict/batch', json={'customers': [VALID]}), 200)

# Validation
check("Invalid gender",
      requests.post(f'{BASE}/predict', json={**VALID, 'gender': 'X'}), 422)
check("Negative tenure",
      requests.post(f'{BASE}/predict', json={**VALID, 'tenure': -1}), 422)
check("Zero charges",
      requests.post(f'{BASE}/predict', json={**VALID, 'MonthlyCharges': 0}), 422)
check("Empty batch",
      requests.post(f'{BASE}/predict/batch', json={'customers': []}), 422)
check("Missing field",
      requests.post(f'{BASE}/predict',
                    json={k:v for k,v in VALID.items() if k != 'tenure'}), 422)

# Edge cases
check("Tenure=0",
      requests.post(f'{BASE}/predict',
                    json={**VALID, 'tenure': 0, 'TotalCharges': 0}), 200)
check("Tenure=120",
      requests.post(f'{BASE}/predict',
                    json={**VALID, 'tenure': 120, 'TotalCharges': 10260}), 200)

print()

# Verify prediction values
r = requests.post(f'{BASE}/predict', json=VALID).json()
print(f"High-risk prediction:")
print(f"  probability : {r['churn_probability']}")
print(f"  prediction  : {r['prediction']}")
print(f"  risk_tier   : {r['risk_tier']}")
print(f"  reason      : {r['reason'][:80]}...")
# \# Customer Churn Predictor

# 

# An end-to-end ML pipeline that predicts telecom customer churn,

# deployed as a REST API with a live monitoring dashboard.

# 

# \## Stack

# Python · scikit-learn · XGBoost · MLflow · FastAPI · Docker · Streamlit

# 

# \## Results

# \*(to be filled in after training)\*

# 

# \## Live Demo

# \- API: \*(link after deployment)\*

# \- Dashboard: \*(link after deployment)\*

# 

# \## Getting Started

# \*(to be filled in)\*

# 

# \## What I'd improve with more time

# \*(to be filled in)\*

## API

The churn prediction model is served as a REST API built with FastAPI.
## Live Demo

- **API** : https://churn-predictor-api.onrender.com
- **Docs** : https://churn-predictor-api.onrender.com/docs

### Running locally

```bash
uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/model-info` | Model version and performance metrics |
| POST | `/predict` | Score a single customer |
| POST | `/predict/batch` | Score up to 1000 customers |
| POST | `/explain` | Score with feature-level explanation |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 171.00,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
  }'
```

### Example response

```json
{
  "churn_probability": 0.7823,
  "prediction": 1,
  "risk_tier": "High",
  "threshold_used": 0.35,
  "reason": "Customer has only been with the company for 2 months — new customers have the highest churn risk."
}
```
"""
api/routers/predict.py
──────────────────────
Prediction endpoints for the churn prediction API.

POST /predict        — score a single customer
POST /predict/batch  — score a list of customers (1-1000)

Both endpoints:
- Validate input via Pydantic schemas (CustomerFeatures)
- Apply the same feature engineering used during training
- Use the pre-loaded pipeline from api/model.py
- Apply the optimal decision threshold from threshold_config.json
- Return structured PredictionResponse with probability and risk tier
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

import api.model as model_module
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionResponse,
)

router = APIRouter(tags=['Predictions'])


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as src/features.py engineer_features().

    Must be called on every incoming request before passing to the pipeline.
    The pipeline's preprocessor was fitted on engineered features — passing
    raw features would produce incorrect scaled values and wrong predictions.

    Features added
    --------------
    charges_per_month : TotalCharges / (tenure + 1)
    num_services      : count of 'Yes' values across 6 add-on service columns
    """
    df = df.copy()

    df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    df['num_services'] = (
        df[service_cols]
        .apply(lambda col: (col == 'Yes').astype(int))
        .sum(axis=1)
    )

    return df


def customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convert a CustomerFeatures Pydantic model to a single-row DataFrame
    with the same column names, dtypes, and encodings the pipeline expects.

    Applies the same transformations as src/features.clean_data():
    - Encodes gender: Male=1, Female=0
    - Encodes Yes/No binary columns: Yes=1, No=0
    - Adds derived features: charges_per_month and num_services

    Does NOT encode the target (Churn) — not present in API requests.
    Does NOT apply StandardScaler or OneHotEncoder — those are inside
    the sklearn Pipeline and run automatically on pipeline.predict_proba().
    """
    raw = customer.model_dump()
    df  = pd.DataFrame([raw])

    # ── Apply same binary encoding as src/features.clean_data() ──────────────

    # Encode gender
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Encode Yes/No binary columns
    yes_no_map = {'Yes': 1, 'No': 0}
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map(yes_no_map)

    # SeniorCitizen is already int from the Pydantic schema — no encoding needed

    # ── Add derived features ──────────────────────────────────────────────────
    df = engineer_features_single(df)

    return df


def assign_risk_tier(probability: float, threshold: float) -> str:
    """
    Assign a human-readable risk tier based on churn probability.

    Tiers
    -----
    High   : probability >= threshold + 0.15
             Customer is very likely to churn — priority for retention
    Medium : threshold <= probability < threshold + 0.15
             Customer is at risk — monitor and consider lighter intervention
    Low    : probability < threshold
             Customer is likely to stay — no action needed

    Parameters
    ----------
    probability : float — model's predicted churn probability
    threshold   : float — optimal decision threshold from threshold_config.json
    """
    if probability >= threshold + 0.15:
        return 'High'
    elif probability >= threshold:
        return 'Medium'
    else:
        return 'Low'


def make_prediction(customer: CustomerFeatures) -> PredictionResponse:
    """
    Core prediction logic — shared by single and batch endpoints.

    Parameters
    ----------
    customer : validated CustomerFeatures Pydantic model

    Returns
    -------
    PredictionResponse with probability, binary prediction, and risk tier
    """
    if not model_module.is_ready():
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Service is not ready.',
        )

    pipeline  = model_module.get_pipeline()
    threshold = model_module.get_threshold()

    # Convert to DataFrame with engineered features
    df = customer_to_dataframe(customer)

    # Get churn probability
    probability = float(pipeline.predict_proba(df)[0, 1])

    # Apply threshold
    prediction = int(probability >= threshold)

    # Assign risk tier
    risk_tier = assign_risk_tier(probability, threshold)

    return PredictionResponse(
        churn_probability = round(probability, 4),
        prediction        = prediction,
        risk_tier         = risk_tier,
        threshold_used    = threshold,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    '/predict',
    response_model=PredictionResponse,
    summary='Predict churn for a single customer',
    description=(
        'Returns the churn probability, binary prediction (0/1), '
        'and risk tier (High/Medium/Low) for one customer.'
    ),
)
def predict_single(customer: CustomerFeatures):
    """
    Score a single customer for churn risk.

    The model applies the optimal decision threshold determined during
    training. Customers above the threshold are classified as churners.

    Risk tiers:
    - High   : probability >= threshold + 0.15
    - Medium : threshold <= probability < threshold + 0.15
    - Low    : probability < threshold
    """
    return make_prediction(customer)


@router.post(
    '/predict/batch',
    response_model=BatchPredictionResponse,
    summary='Predict churn for a batch of customers',
    description=(
        'Score 1-1000 customers in a single request. '
        'Returns individual predictions plus a summary of risk tier distribution.'
    ),
)
def predict_batch(request: BatchPredictionRequest):
    """
    Score a batch of customers for churn risk.

    Accepts 1-1000 customers per request. Each customer is scored
    independently. Returns per-customer predictions plus aggregate
    risk tier counts for quick dashboard display.
    """
    if not model_module.is_ready():
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Service is not ready.',
        )

    predictions = [make_prediction(customer)
                   for customer in request.customers]

    high_risk   = sum(1 for p in predictions if p.risk_tier == 'High')
    medium_risk = sum(1 for p in predictions if p.risk_tier == 'Medium')
    low_risk    = sum(1 for p in predictions if p.risk_tier == 'Low')

    return BatchPredictionResponse(
        predictions = predictions,
        total       = len(predictions),
        high_risk   = high_risk,
        medium_risk = medium_risk,
        low_risk    = low_risk,
    )
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

import numpy as np
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ExplanationResponse,
    FeatureContribution,
    PredictionResponse,
)
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

def generate_reason(df: pd.DataFrame, probability: float,
                    threshold: float) -> str:
    """
    Generate a plain-English sentence explaining the prediction.

    Uses simple rule-based logic on the raw feature values — not SHAP.
    Fast, deterministic, and requires no additional computation.

    For the full SHAP-based explanation use GET /explain.

    Parameters
    ----------
    df          : single-row DataFrame with engineered features
    probability : predicted churn probability
    threshold   : decision threshold

    Returns
    -------
    str — one sentence explaining the primary risk factor or protective factor
    """
    row = df.iloc[0]

    if probability >= threshold:
        # High risk — identify the biggest risk factor
        if row['tenure'] <= 6:
            return (f"Customer has only been with the company for "
                    f"{int(row['tenure'])} months — new customers have "
                    f"the highest churn risk.")
        elif row['Contract'] == 'Month-to-month':
            return ("Customer is on a month-to-month contract with no "
                    "lock-in — the single strongest predictor of churn.")
        elif row['MonthlyCharges'] >= 80:
            return (f"Customer's monthly bill of ${row['MonthlyCharges']:.0f} "
                    f"is high — elevated charges are a significant churn driver.")
        elif row['InternetService'] == 'Fiber optic' and row['num_services'] <= 1:
            return ("Fiber optic customer with few add-on services — "
                    "high cost, low switching cost.")
        else:
            return (f"Multiple risk factors present — churn probability "
                    f"is {probability:.0%}.")
    else:
        # Low risk — identify the strongest protective factor
        if row['Contract'] == 'Two year':
            return ("Customer is on a two-year contract — the strongest "
                    "protective factor against churn.")
        elif row['tenure'] >= 36:
            return (f"Customer has been with the company for "
                    f"{int(row['tenure'])} months — long tenure is "
                    f"strongly protective.")
        elif row['num_services'] >= 4:
            return (f"Customer subscribes to {int(row['num_services'])} "
                    f"add-on services — high switching cost reduces churn risk.")
        else:
            return (f"Customer profile shows low churn risk "
                    f"(probability {probability:.0%}).")

def make_prediction(customer: CustomerFeatures) -> PredictionResponse:
    """
    Core prediction logic — shared by single and batch endpoints.
    """
    if not model_module.is_ready():
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Service is not ready.',
        )

    pipeline  = model_module.get_pipeline()
    threshold = model_module.get_threshold()

    df          = customer_to_dataframe(customer)
    probability = float(pipeline.predict_proba(df)[0, 1])
    prediction  = int(probability >= threshold)
    risk_tier   = assign_risk_tier(probability, threshold)
    reason      = generate_reason(df, probability, threshold)

    return PredictionResponse(
        churn_probability = round(probability, 4),
        prediction        = prediction,
        risk_tier         = risk_tier,
        threshold_used    = threshold,
        reason            = reason,
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

@router.post(
    '/explain',
    response_model=ExplanationResponse,
    summary='Predict with feature-level explanation',
    description=(
        'Returns the churn prediction plus the top features '
        'driving the risk score up or down.'
    ),
)
def explain_prediction(customer: CustomerFeatures):
    """
    Score a customer and return a feature-level explanation.

    Uses the model's feature importances weighted by the customer's
    feature values to approximate which features most influenced
    this specific prediction.

    For exact SHAP values, use the SHAP analysis notebook.
    This endpoint provides a fast approximation suitable for
    real-time dashboard display.
    """
    if not model_module.is_ready():
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Service is not ready.',
        )

    pipeline  = model_module.get_pipeline()
    threshold = model_module.get_threshold()

    df          = customer_to_dataframe(customer)
    probability = float(pipeline.predict_proba(df)[0, 1])
    prediction  = int(probability >= threshold)
    risk_tier   = assign_risk_tier(probability, threshold)
    reason      = generate_reason(df, probability, threshold)

    # ── Compute feature contributions ─────────────────────────────────────────
    preprocessor   = pipeline.named_steps['preprocessor']
    model          = pipeline.named_steps['model']
    X_transformed  = preprocessor.transform(df)

    # Get feature names
    num_names = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'charges_per_month', 'num_services',
    ]
    try:
        cat_names = (preprocessor
                     .named_transformers_['cat']
                     .get_feature_names_out()
                     .tolist())
    except Exception:
        cat_names = []

    bin_names = [
        'gender', 'Partner', 'Dependents',
        'PhoneService', 'PaperlessBilling', 'SeniorCitizen',
    ]
    feature_names = num_names + cat_names + bin_names

    # Use model feature importances × transformed feature values
    # as a fast proxy for SHAP values
    importances    = model.feature_importances_
    feature_values = X_transformed[0]

    # Weighted contributions — importance × |feature value|
    contributions = importances * np.abs(feature_values)

    # Pair with names, sort by contribution magnitude
    feature_contribs = sorted(
        zip(feature_names[:len(contributions)],
            feature_values[:len(contributions)],
            contributions),
        key=lambda x: x[2],
        reverse=True,
    )

    # Split into risk-increasing and risk-decreasing
    # Positive scaled value + high importance = increases risk
    # Negative scaled value + high importance = decreases risk
    risk_factors   = []
    protective     = []

    for name, val, contrib in feature_contribs:
        if len(risk_factors) >= 3 and len(protective) >= 3:
            break
        direction = 'increases' if val > 0 else 'decreases'
        entry = FeatureContribution(
            feature   = name,
            value     = round(float(val), 4),
            direction = direction,
        )
        if direction == 'increases' and len(risk_factors) < 3:
            risk_factors.append(entry)
        elif direction == 'decreases' and len(protective) < 3:
            protective.append(entry)

    # Baseline ≈ training set churn rate
    baseline = 0.265

    return ExplanationResponse(
        churn_probability   = round(probability, 4),
        prediction          = prediction,
        risk_tier           = risk_tier,
        threshold_used      = threshold,
        reason              = reason,
        top_risk_factors    = risk_factors,
        top_protective      = protective,
        baseline_probability= baseline,
    )
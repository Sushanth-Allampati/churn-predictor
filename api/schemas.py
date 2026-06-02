"""
api/schemas.py
──────────────
Pydantic request and response schemas for the churn prediction API.

All input validation happens here — FastAPI automatically returns
422 Unprocessable Entity with detailed error messages when validation fails.
The /predict endpoint never receives invalid data.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status     : str = Field(..., description="'ok' if service is running")
    model_loaded: bool = Field(..., description="True if model is loaded and ready")
    version    : str = Field(..., description="API version string")


class ModelInfoResponse(BaseModel):
    """Response schema for GET /model-info."""
    model_name       : str
    model_version    : str
    optimal_threshold: float
    calibration      : str
    test_roc_auc     : float
    test_pr_auc      : float
    test_f1          : float
    test_precision   : float
    test_recall      : float
    brier_score      : float


class CustomerFeatures(BaseModel):
    """
    Input schema for POST /predict.

    All 19 features required for the churn prediction model.
    Field validators catch invalid values before they reach the model.

    Notes
    -----
    - SeniorCitizen is int (0 or 1) in the raw data — kept as int here
    - TotalCharges can be 0 for new customers (tenure=0)
    - All Yes/No string fields are validated to 'Yes' or 'No' exactly
    """

    # ── Customer demographics ─────────────────────────────────────────────────
    gender         : str  = Field(..., description="'Male' or 'Female'")
    SeniorCitizen  : int  = Field(..., ge=0, le=1,
                                  description="1 if senior citizen, 0 otherwise")
    Partner        : str  = Field(..., description="'Yes' or 'No'")
    Dependents     : str  = Field(..., description="'Yes' or 'No'")

    # ── Account info ──────────────────────────────────────────────────────────
    tenure         : int  = Field(..., ge=0, le=120,
                                  description="Months as customer (0-120)")
    Contract       : str  = Field(...,
                                  description="'Month-to-month', 'One year', or 'Two year'")
    PaperlessBilling: str = Field(..., description="'Yes' or 'No'")
    PaymentMethod  : str  = Field(...,
                                  description="'Electronic check', 'Mailed check', "
                                              "'Bank transfer (automatic)', or "
                                              "'Credit card (automatic)'")
    MonthlyCharges : float = Field(..., gt=0,
                                   description="Monthly bill in USD (must be > 0)")
    TotalCharges   : float = Field(..., ge=0,
                                   description="Total spend to date in USD (>= 0)")

    # ── Phone service ─────────────────────────────────────────────────────────
    PhoneService   : str  = Field(..., description="'Yes' or 'No'")
    MultipleLines  : str  = Field(...,
                                  description="'Yes', 'No', or 'No phone service'")

    # ── Internet service ──────────────────────────────────────────────────────
    InternetService: str  = Field(...,
                                  description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity : str  = Field(...,
                                  description="'Yes', 'No', or 'No internet service'")
    OnlineBackup   : str  = Field(...,
                                  description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(...,
                                  description="'Yes', 'No', or 'No internet service'")
    TechSupport    : str  = Field(...,
                                  description="'Yes', 'No', or 'No internet service'")
    StreamingTV    : str  = Field(...,
                                  description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str  = Field(...,
                                  description="'Yes', 'No', or 'No internet service'")

    # ── Field validators ──────────────────────────────────────────────────────

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in {'Male', 'Female'}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    @classmethod
    def validate_yes_no(cls, v):
        if v not in {'Yes', 'No'}:
            raise ValueError(f"Must be 'Yes' or 'No', got '{v}'")
        return v

    @field_validator('Contract')
    @classmethod
    def validate_contract(cls, v):
        valid = {'Month-to-month', 'One year', 'Two year'}
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    @field_validator('InternetService')
    @classmethod
    def validate_internet(cls, v):
        valid = {'DSL', 'Fiber optic', 'No'}
        if v not in valid:
            raise ValueError(f"InternetService must be one of {valid}")
        return v

    @field_validator('PaymentMethod')
    @classmethod
    def validate_payment(cls, v):
        valid = {
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)',
        }
        if v not in valid:
            raise ValueError(f"PaymentMethod must be one of {valid}")
        return v

    @field_validator('MultipleLines')
    @classmethod
    def validate_multiple_lines(cls, v):
        valid = {'Yes', 'No', 'No phone service'}
        if v not in valid:
            raise ValueError(f"MultipleLines must be one of {valid}")
        return v

    @field_validator('OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies')
    @classmethod
    def validate_internet_service_features(cls, v):
        valid = {'Yes', 'No', 'No internet service'}
        if v not in valid:
            raise ValueError(f"Must be 'Yes', 'No', or 'No internet service'")
        return v

    model_config = {'json_schema_extra': {
        'example': {
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
    }}


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""
    churn_probability : float = Field(...,
                                      description="Probability of churn (0-1)")
    prediction        : int   = Field(...,
                                      description="Binary prediction (1=Churn, 0=No Churn)")
    risk_tier         : str   = Field(...,
                                      description="'High', 'Medium', or 'Low'")
    threshold_used    : float = Field(...,
                                      description="Decision threshold applied")


class BatchPredictionRequest(BaseModel):
    """Request schema for POST /predict/batch."""
    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of 1-1000 customers to score",
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for POST /predict/batch."""
    predictions : list[PredictionResponse]
    total       : int
    high_risk   : int
    medium_risk : int
    low_risk    : int
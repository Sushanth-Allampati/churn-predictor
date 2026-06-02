"""
api/routers/health.py
─────────────────────
Health and model-info endpoints.

GET /health     — liveness check used by deployment platforms
GET /model-info — model metadata for monitoring and debugging
"""

from fastapi import APIRouter, HTTPException
from api.schemas import HealthResponse, ModelInfoResponse
import api.model as model_module

router = APIRouter(tags=['Health'])


@router.get(
    '/health',
    response_model=HealthResponse,
    summary='Liveness check',
    description='Returns 200 OK if the service is running. '
                'Returns 503 if the model failed to load.',
)
def health_check():
    """
    Liveness endpoint used by deployment platforms (Render, GCP Cloud Run)
    to verify the service is running and the model is loaded.

    Returns 503 Service Unavailable if the model is not ready —
    this causes the deployment platform to restart the container.
    """
    if not model_module.is_ready():
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Service is not ready.',
        )

    return HealthResponse(
        status      = 'ok',
        model_loaded= True,
        version     = '1.0.0',
    )


@router.get(
    '/model-info',
    response_model=ModelInfoResponse,
    summary='Model metadata',
    description='Returns model version, performance metrics, and configuration.',
)
def model_info():
    """
    Returns metadata about the currently loaded model.
    Used by monitoring dashboards and for debugging.
    """
    if not model_module.is_ready():
        raise HTTPException(
            status_code=503,
            detail='Model not loaded.',
        )

    tc = model_module.threshold_config
    tm = model_module.test_metrics

    return ModelInfoResponse(
        model_name        = 'churn-model',
        model_version     = str(model_module.model_version or 'unknown'),
        optimal_threshold = tc.get('optimal_threshold', 0.5),
        calibration       = tc.get('calibration_method', 'none'),
        test_roc_auc      = tm.get('roc_auc', 0.0),
        test_pr_auc       = tm.get('pr_auc',  0.0),
        test_f1           = tm.get('f1',      0.0),
        test_precision    = tm.get('precision', 0.0),
        test_recall       = tm.get('recall',   0.0),
        brier_score       = tm.get('brier',    0.0),
    )
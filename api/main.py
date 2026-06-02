"""
api/main.py
───────────
FastAPI application entry point for the Telco Churn Prediction API.

The app uses FastAPI's lifespan context manager to load the model
once at startup and clean up on shutdown. All prediction endpoints
use the pre-loaded model — no per-request loading.

Running locally
---------------
    uvicorn api.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs    ← interactive Swagger UI
    http://localhost:8000/redoc  ← ReDoc documentation
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import api.model as model_module
from api.routers import health


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Churn Prediction API...")
    try:
        model_module.load_model()
        print("Startup complete.")
    except Exception as e:
        import traceback
        print(f"STARTUP ERROR: {e}")
        traceback.print_exc()
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = 'Telco Customer Churn Prediction API',
    description = (
        'Predicts the probability of a telecom customer churning '
        'based on their account and service features. '
        'Built with LightGBM, scikit-learn, and FastAPI.'
    ),
    version     = '1.0.0',
    lifespan    = lifespan,
    docs_url    = '/docs',
    redoc_url   = '/redoc',
)


# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ['*'],   # tighten this in production
    allow_methods  = ['*'],
    allow_headers  = ['*'],
)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
# predict router added on Day 22


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
def root():
    """Redirect root to docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url='/docs')
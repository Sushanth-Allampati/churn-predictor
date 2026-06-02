"""
api/main.py
───────────
FastAPI application entry point for the Telco Churn Prediction API.
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import api.model as model_module
from api.routers import health, predict

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s | %(levelname)s | %(message)s',
    datefmt= '%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Churn Prediction API...")
    try:
        model_module.load_model()
        logger.info("Startup complete. API is ready.")
    except Exception as e:
        import traceback
        logger.error(f"STARTUP ERROR: {e}")
        traceback.print_exc()
    yield
    logger.info("Shutting down Churn Prediction API.")


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


# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ['*'],
    allow_methods  = ['*'],
    allow_headers  = ['*'],
)


@app.middleware('http')
async def log_requests(request: Request, call_next):
    """
    Log every request with method, path, status code, and duration.
    This middleware runs for every request including health checks.
    """
    start   = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000   # ms

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration:.1f}ms)"
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exception and return a clean JSON error response.
    Without this, unhandled exceptions return an HTML 500 page.
    """
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'error'  : 'Internal server error',
            'detail' : str(exc),
            'path'   : str(request.url.path),
        }
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(predict.router)


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url='/docs')
"""FastAPI application for the Financial Risk Prediction Platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import explain, models, predict
from src.config import API_VERSION, MODEL_VERSION, SERVICE_NAME
from src.models.predictor import RiskRuntime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the champion pipeline once and keep a small explanation cache."""

    app.state.runtime = RiskRuntime.load()
    app.state.scored = {}
    yield


app = FastAPI(
    title="Financial Risk Prediction API",
    version=API_VERSION,
    description=(
        "Real-time credit default scoring with a cost-sensitive threshold, "
        "SHAP feature attributions, and a five-model leaderboard."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(models.router)


@app.get("/health")
def health() -> dict[str, object]:
    """Report whether the champion pipeline loaded."""

    runtime: RiskRuntime = app.state.runtime
    return {
        "status": "healthy" if runtime.ready else "degraded",
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "active_model": runtime.model_name if runtime.ready else None,
        "champion_model": runtime.model_name if runtime.ready else None,
        "model_version": MODEL_VERSION if runtime.ready else None,
        "pipeline_loaded": runtime.ready,
        "model_loaded": runtime.ready,
        "detail": runtime.error,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

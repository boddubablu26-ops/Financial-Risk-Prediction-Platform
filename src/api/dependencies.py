"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from src.models.predictor import RiskRuntime


def get_runtime(request: Request) -> RiskRuntime:
    """Return the champion runtime stored on application startup.

    Args:
        request: Incoming request.

    Returns:
        Loaded runtime.

    Raises:
        HTTPException: 503 when training artifacts are missing.
    """

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not runtime.ready:
        detail = getattr(runtime, "error", None) or "Model pipeline is not loaded."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return runtime


def get_runtime_optional(request: Request) -> RiskRuntime | None:
    """Return the runtime even when the model is not loaded."""

    return getattr(request.app.state, "runtime", None)
